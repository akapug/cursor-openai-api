/**
 * Local OpenAI-compatible proxy that translates requests to Cursor's gRPC protocol.
 *
 * Accepts POST /v1/chat/completions in OpenAI format, translates to Cursor's
 * protobuf/HTTP2 Connect protocol, and streams back OpenAI-format SSE.
 *
 * Tool calling uses Cursor's native MCP tool protocol:
 * - OpenAI tool defs → McpToolDefinition in RequestContext
 * - Cursor toolCallStarted/Delta/Completed → OpenAI tool_calls SSE chunks
 * - mcpArgs exec → pause stream, return tool_calls to caller
 * - Follow-up request with tool results → resume bridge with mcpResult
 *
 * HTTP/2 transport is delegated to a Node child process (h2-bridge.mjs)
 * because Bun's node:http2 module is broken.
 */
import { create, fromBinary, fromJson, type JsonValue, toBinary, toJson } from "@bufbuild/protobuf";
import { CURSOR_CLIENT_VERSION, getCursorModels } from "./models";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { BinaryWriter, WireType } from "@bufbuild/protobuf/wire";
import {
  AgentClientMessageSchema,
  AgentRunRequestSchema,
  AgentServerMessageSchema,
  ClientHeartbeatSchema,
  ConversationActionSchema,
  ConversationStateStructureSchema,
  BackgroundShellSpawnResultSchema,
  DeleteResultSchema,
  DeleteRejectedSchema,
  DiagnosticsResultSchema,
  ExecClientMessageSchema,
  FetchErrorSchema,
  FetchResultSchema,
  GetBlobResultSchema,
  GrepErrorSchema,
  GrepResultSchema,
  KvClientMessageSchema,
  LsRejectedSchema,
  LsResultSchema,
  McpErrorSchema,
  McpResultSchema,
  McpSuccessSchema,
  McpTextContentSchema,
  McpToolDefinitionSchema,
  McpToolNotFoundSchema,
  McpToolResultContentItemSchema,
  ModelDetailsSchema,
  ReadRejectedSchema,
  ReadResultSchema,
  RequestContextResultSchema,
  RequestContextSchema,
  RequestContextSuccessSchema,
  SetBlobResultSchema,
  ShellRejectedSchema,
  ShellResultSchema,
  ShellStreamSchema,
  UserMessageActionSchema,
  UserMessageSchema,
  WriteRejectedSchema,
  WriteResultSchema,
  WriteShellStdinErrorSchema,
  WriteShellStdinResultSchema,
  type AgentServerMessage,
  type ExecServerMessage,
  type KvServerMessage,
  type McpToolDefinition,
} from "./proto/agent_pb";
import { createHash } from "node:crypto";
import { resolve as pathResolve } from "node:path";

const CURSOR_API_URL = "https://api2.cursor.sh";
const CONNECT_END_STREAM_FLAG = 0b00000010;
const BRIDGE_PATH = pathResolve(import.meta.dir, "h2-bridge.mjs");

// helm patch (2026-09-25): CURSOR_BRIDGE_TRACE=1 logs every server frame's
// case to stderr, so a stall can be read off the log instead of guessed at.
const TRACE = process.env.CURSOR_BRIDGE_TRACE === "1";
function trace(...parts: unknown[]): void {
  if (TRACE) console.error("[trace]", new Date().toISOString().slice(11, 23), ...parts);
}

/**
 * Render protobuf wire bytes whose schema we do not have: field numbers,
 * nested messages where the bytes parse as one, short strings otherwise.
 * Used when Cursor sends a message kind newer than the vendored proto, so the
 * log names WHAT arrived rather than "undefined".
 */
function wireDump(bytes: Uint8Array, depth = 0): string {
  const parts: string[] = [];
  let i = 0;
  const varint = (): number => {
    let v = 0, shift = 0, b = 0;
    do {
      if (i >= bytes.length) throw new Error("truncated varint");
      b = bytes[i++]!;
      v += (b & 0x7f) * 2 ** shift;
      shift += 7;
    } while (b & 0x80);
    return v;
  };
  try {
    while (i < bytes.length) {
      const tag = varint();
      const no = Math.floor(tag / 8), wt = tag % 8;
      if (no === 0) throw new Error("field 0");
      if (wt === 0) parts.push(`${no}=${varint()}`);
      else if (wt === 1) { i += 8; parts.push(`${no}:f64`); }
      else if (wt === 5) { i += 4; parts.push(`${no}:f32`); }
      else if (wt === 2) {
        const len = varint();
        if (i + len > bytes.length) throw new Error("truncated len");
        const sub = bytes.subarray(i, i + len);
        i += len;
        const nested = depth < 4 && len > 0 ? wireDump(sub, depth + 1) : "";
        const text = new TextDecoder("utf-8", { fatal: false }).decode(sub);
        parts.push(nested && !nested.startsWith("!")
          ? `${no}{${nested}}`
          : `${no}"${text.slice(0, 80).replace(/\s+/g, " ")}"`);
      } else throw new Error(`wire type ${wt}`);
    }
  } catch (err) {
    return `!${err instanceof Error ? err.message : String(err)}`;
  }
  return parts.join(" ");
}

/**
 * The unknown fields protobuf-es kept on a decoded message, rendered. Its
 * `data` is everything AFTER the tag, so the tag is re-encoded in front and
 * the field is read back as a one-field message.
 */
function unknownFields(msg: { $unknown?: { no: number; wireType: number; data: Uint8Array }[] }): string {
  return (msg.$unknown ?? []).map((u) => {
    const tag: number[] = [];
    for (let t = u.no * 8 + u.wireType; ; t = Math.floor(t / 128)) {
      if (t < 128) { tag.push(t); break; }
      tag.push((t % 128) | 0x80);
    }
    return wireDump(Buffer.concat([Buffer.from(tag), u.data]));
  }).join(" | ") || "none";
}

// --- Types ---

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

interface OpenAIMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
}

interface OpenAIToolDef {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface ChatCompletionRequest {
  model: string;
  messages: OpenAIMessage[];
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  tools?: OpenAIToolDef[];
  tool_choice?: unknown;
}

interface CursorRequestPayload {
  requestBytes: Uint8Array;
  blobStore: Map<string, Uint8Array>;
  mcpTools: McpToolDefinition[];
  /** The client conversation this request carries (`conversationStamp`). */
  conversation?: ConversationStamp;
}

/**
 * helm patch: WHICH CLIENT CONVERSATION A RUN HOLDS. A paused run is found by
 * the tool-call ids it handed out (findPausedRun), and that alone cannot tell
 * a continuation from a REWRITE: Claude Code's autocompact keeps the last
 * tool call and its result after the summary, so the request after a
 * compaction answers the paused run's tool call while carrying a different,
 * much shorter conversation. The stamp is the first user message's digest
 * and the message count; a continuation keeps the head and grows.
 */
interface ConversationStamp {
  head: string;
  seen: number;
}

/** A pending tool execution waiting for results from the caller. */
interface PendingExec {
  execId: string;
  execMsgId: number;
  /** The id the CLIENT sees and sends back: [A-Za-z0-9_-] only. */
  toolCallId: string;
  /** Cursor's own id for the call (may carry "\n"); kept for logs. */
  cursorToolCallId: string;
  toolName: string;
  /** Decoded arguments JSON string for SSE tool_calls emission. */
  decodedArgs: string;
}

/** The h2 child that carries one live Cursor stream. */
type Bridge = ReturnType<typeof spawnBridge>;

/**
 * One live Cursor agent run, kept across the client's tool round trips.
 *
 * helm patch (2026-09-25): A RUN IS FOUND BY THE TOOL-CALL IDS IT HANDED OUT,
 * NEVER BY WHAT THE CONVERSATION LOOKS LIKE. The old key was a hash of the
 * model and the first 200 characters of the first user message. Claude Code
 * opens every conversation — the seat's own and each subagent's — with the
 * same <system-reminder> preamble, so a seat and all of its subagents shared
 * ONE key. A subagent's tool call overwrote the seat's paused bridge in the
 * map (the seat's bridge leaked, heartbeating on); the seat's tool result was
 * then fed to the SUBAGENT's run and its answer streamed to the wrong
 * conversation; and any fresh request under the same key ended whichever run
 * was paused there. A tool-call id is unique per call and comes back verbatim
 * with its result, so it is the real identity of "the run this result is for".
 */
interface Run {
  /** Serial for log lines. */
  id: number;
  modelId: string;
  bridge: Bridge;
  heartbeatTimer: NodeJS.Timeout;
  blobStore: Map<string, Uint8Array>;
  mcpTools: McpToolDefinition[];
  /** Tool calls handed to the client whose results have not come back. */
  awaiting: PendingExec[];
  /** Tool calls Cursor asked for while no client request was attached. */
  queued: PendingExec[];
  /** Text/reasoning that arrived while no client request was attached. */
  held: Array<{ text: string; thinking: boolean }>;
  /** The client response currently attached, if any. */
  sink: Sink | null;
  /** Set by an exec mid-chunk; the pause happens once the chunk is read. */
  pauseAfterChunk: boolean;
  /** Undecoded bytes carried to the next chunk. */
  pending: Buffer;
  closed: boolean;
  /** Ends a paused run nobody came back for. */
  expiry: NodeJS.Timeout | null;
  /** Ends an attached run that is stuck (see `stallDue`). */
  stall: NodeJS.Timeout | null;
  /**
   * Cursor's own call ids of the tool calls it opened (toolCallStarted) and
   * has not completed (toolCallCompleted). Non-empty means Cursor waits on
   * the bridge's answer to an exec, so its heartbeats are no progress.
   */
  openTools: Set<string>;
  /** When the last byte from Cursor arrived, a heartbeat included (performance.now ms). */
  lastByteAt: number;
  /** When the last frame that is not a heartbeat arrived (ms). */
  lastProgressAt: number;
  /** When the attached client last got a byte of its response (ms). */
  lastSentAt: number;
  /**
   * The size of the request now attached, estimated from its messages and
   * tools (`requestTokens`): set when the run starts and replaced by every
   * request that resumes it. This is the prompt count a response reports.
   */
  promptTokens: number;
  /**
   * Cursor's own count of its conversation during the response now attached
   * (0 = none yet), reset when a paused run resumes. It is traced beside the
   * usage and never reported: it is not the conversation the client sent
   * (usageOf).
   */
  contextTokens: number;
  /** The client conversation this run holds; absent = unknown, not vetoed. */
  conversation?: ConversationStamp;
  /** Output tokens Cursor reported while no client was attached (traced only). */
  heldOutputTokens: number;
  /**
   * The Connect error Cursor ended this run with, once one arrived. The
   * response it lands in ends with an ERROR the client retries, never with
   * the error as reply text (see `fail`).
   */
  failure: Error | null;
  /** Set while a resumed run waits for its first frame; called once it comes. */
  opening: (() => void) | null;
}

/** Paused runs by the client-facing id of every tool call they await. */
const pausedRuns = new Map<string, Run>();
let runSerial = 0;

/**
 * helm patch (2026-09-27): THE CLIENT'S IDLE TIMEOUT. Bun.serve closes a
 * connection that has carried no byte for this long, and 255 s is the most it
 * allows. MEASURED 2026-09-25: a turn stuck on an exec answer that never
 * reached Cursor was cut at 254 s as "unexpected EOF". Every watchdog window
 * ends a run before this does, so the client is told why in words.
 */
const IDLE_TIMEOUT_S = 255;
/** A fresh or resumed request may wait this long before its response attaches. */
const FIRST_FRAME_MS = 15_000;
/** Scheduling room after the watchdog speaks and before Bun may cut the request. */
const IDLE_MARGIN_MS = 15_000;
/**
 * The attached response's last safe window. Bun's clock already ran through
 * the first-frame wait, so both that wait and a real scheduling margin must
 * fit before its 255 s limit.
 */
const WINDOW_CEILING_MS = IDLE_TIMEOUT_S * 1000
  - FIRST_FRAME_MS - IDLE_MARGIN_MS;
/** Cursor heartbeats every 10 s; a window under three periods misreads a live stream. */
const STALL_FLOOR_MS = 30_000;
const STALL_DEFAULT_MS = 90_000;

/**
 * The stall window from CURSOR_BRIDGE_STALL_MS: a whole number of
 * milliseconds, held between STALL_FLOOR_MS and WINDOW_CEILING_MS. Unset or
 * empty is the default; anything else that is not a whole number is the
 * default too, and says so on stderr, so a typo never silently moves it.
 */
function stallWindow(
  raw: string | undefined,
  warn: (message: string) => void = (m) => console.error(m),
): number {
  const text = (raw ?? "").trim();
  if (!text) return STALL_DEFAULT_MS;
  const ms = /^\d+$/.test(text) ? Number(text) : NaN;
  if (!Number.isSafeInteger(ms)) {
    warn(`[bridge] CURSOR_BRIDGE_STALL_MS=${JSON.stringify(raw)} is not a whole number of milliseconds; the stall window is the ${STALL_DEFAULT_MS} ms default`);
    return STALL_DEFAULT_MS;
  }
  const held = Math.min(Math.max(ms, STALL_FLOOR_MS), WINDOW_CEILING_MS);
  if (held !== ms) {
    warn(`[bridge] CURSOR_BRIDGE_STALL_MS=${ms} is outside ${STALL_FLOOR_MS}..${WINDOW_CEILING_MS} ms (three of Cursor's 10 s heartbeat periods, with the ${FIRST_FRAME_MS} ms first-frame wait and ${IDLE_MARGIN_MS} ms scheduling margin inside the client's idle timeout (${IDLE_TIMEOUT_S} s)); the stall window is ${held} ms`);
  }
  return held;
}

/** Run lifetimes, read at call time (tests shorten them). */
const limits = {
  /**
   * A paused run nobody resumes (the client was interrupted, crashed, or went
   * another way) is ended after this long. Claude Code's own Bash tool caps a
   * command at 10 minutes, so a live tool loop always comes back before it.
   */
  pausedTtlMs: 15 * 60_000,
  /**
   * helm patch (2026-09-25, 2026-09-27): while a client is attached, a run
   * that sends nothing for this long, not even a heartbeat, is a dead stream;
   * and a run with a tool call open that sends only heartbeats for this long
   * is stuck: MEASURED, Cursor heartbeats every 10 s exactly while it waits
   * on an exec answer that never reached it. Heartbeats with NO tool call
   * open are the model working (it can think for minutes), so they keep the
   * run alive (see `stallDue`). CURSOR_BRIDGE_STALL_MS moves it.
   */
  stallMs: stallWindow(process.env.CURSOR_BRIDGE_STALL_MS),
  /**
   * helm patch (2026-09-27): the longest the attached client may get no byte
   * of its response. Past it Bun's idle timeout closes the connection as
   * "unexpected EOF", so the bridge ends the run first and says why.
   */
  idleGuardMs: WINDOW_CEILING_MS,
  /**
   * How long a request waits for the run's first frame before it commits to
   * a 200: a fresh run (peekFirstFrame) and a resumed one (awaitFirstFrame)
   * alike. A refusal inside the wait is a real 429.
   */
  firstFrameMs: FIRST_FRAME_MS,
};

// --- H2 Bridge IPC ---

/** Length-prefix a message: [4-byte BE length][payload] */
function lpEncode(data: Uint8Array): Buffer {
  const buf = Buffer.alloc(4 + data.length);
  buf.writeUInt32BE(data.length, 0);
  buf.set(data, 4);
  return buf;
}

/** Connect protocol frame: [1-byte flags][4-byte BE length][payload] */
function frameConnectMessage(data: Uint8Array, flags = 0): Buffer {
  const frame = Buffer.alloc(5 + data.length);
  frame[0] = flags;
  frame.writeUInt32BE(data.length, 1);
  frame.set(data, 5);
  return frame;
}

/**
 * Spawn the Node H2 bridge and return read/write handles.
 * The bridge uses length-prefixed framing on stdin/stdout.
 */
function spawnBridge(accessToken: string): {
  proc: ReturnType<typeof Bun.spawn>;
  write: (data: Uint8Array) => void;
  end: () => void;
  onData: (cb: (chunk: Buffer) => void) => void;
  onClose: (cb: (code: number) => void) => void;
} {
  const proc = Bun.spawn(["node", BRIDGE_PATH], {
    stdin: "pipe",
    stdout: "pipe",
    // helm patch (2026-09-25): the child says why it died; that line belongs
    // in the bridge log, not in /dev/null.
    stderr: "inherit",
  });

  const config = JSON.stringify({
    accessToken,
    url: CURSOR_API_URL,
    path: "/agent.v1.AgentService/Run",
    clientVersion: CURSOR_CLIENT_VERSION,
  });
  proc.stdin.write(lpEncode(new TextEncoder().encode(config)));

  const cbs = {
    data: null as ((chunk: Buffer) => void) | null,
    close: null as ((code: number) => void) | null,
  };

  (async () => {
    const reader = proc.stdout.getReader();
    let pending = Buffer.alloc(0);

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        pending = Buffer.concat([pending, Buffer.from(value)]);

        while (pending.length >= 4) {
          const len = pending.readUInt32BE(0);
          if (pending.length < 4 + len) break;
          const payload = pending.subarray(4, 4 + len);
          pending = pending.subarray(4 + len);
          cbs.data?.(Buffer.from(payload));
        }
      }
    } catch {
      // Stream ended
    }

    proc.exited.then((code) => cbs.close?.(code ?? 1));
  })();

  return {
    proc,
    write(data) {
      try { proc.stdin.write(lpEncode(data)); } catch {}
    },
    end() {
      try {
        proc.stdin.write(lpEncode(new Uint8Array(0)));
        proc.stdin.end();
      } catch {}
    },
    onData(cb) { cbs.data = cb; },
    onClose(cb) { cbs.close = cb; },
  };
}

// --- Proxy Server ---

let proxyServer: ReturnType<typeof Bun.serve> | undefined;
let proxyPort: number | undefined;

export function getProxyPort(): number | undefined {
  return proxyPort;
}

export async function startProxy(
  getAccessToken: () => Promise<string>,
  port: number = 0,
): Promise<number> {
  if (proxyServer && proxyPort) return proxyPort;

  proxyServer = Bun.serve({
    // helm patch (2026-09-25): LOOPBACK ONLY. The server takes no credential
    // from its callers and spends the owner's Cursor account on every request;
    // bound to 0.0.0.0 (MEASURED: `*:18315` in ss), anyone who could reach the
    // machine could spend it. The only caller is the seat's local proxy.
    hostname: "127.0.0.1",
    port,
    idleTimeout: IDLE_TIMEOUT_S, // max — Cursor responses can take 30s+
    async fetch(req) {
      const url = new URL(req.url);

      if (req.method === "GET" && url.pathname === "/v1/models") {
        const accessToken = await getAccessToken();
        let cursorModels: Awaited<ReturnType<typeof getCursorModels>>;
        try {
          cursorModels = await getCursorModels(accessToken);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          return new Response(
            JSON.stringify({ error: { message, type: "upstream_error", code: "model_discovery_failed" } }),
            { status: 502, headers: { "Content-Type": "application/json" } },
          );
        }
        const openAIModels = cursorModels.map((m) => ({
          id: m.id,
          object: "model",
          created: 0,
          owned_by: "cursor",
          permission: [],
          root: m.id,
          parent: null,
        }));
        return new Response(
          JSON.stringify({ object: "list", data: openAIModels }),
          { headers: { "Content-Type": "application/json" } },
        );
      }

      if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
        try {
          const body = (await req.json()) as ChatCompletionRequest;
          const accessToken = await getAccessToken();
          return handleChatCompletion(body, accessToken);
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          return new Response(
            JSON.stringify({
              error: { message, type: "server_error", code: "internal_error" },
            }),
            { status: 500, headers: { "Content-Type": "application/json" } },
          );
        }
      }

      return new Response("Not Found", { status: 404 });
    },
  });

  proxyPort = proxyServer.port;
  if (!proxyPort) throw new Error("Failed to bind proxy to a port");
  return proxyPort;
}

export function stopProxy(): void {
  if (proxyServer) {
    proxyServer.stop();
    proxyServer = undefined;
    proxyPort = undefined;
  }
  // Clean up any lingering paused runs
  for (const run of new Set(pausedRuns.values())) endRun(run, "proxy stopping");
}

// --- Chat Completion Handler ---

async function handleChatCompletion(
  body: ChatCompletionRequest,
  accessToken: string,
): Promise<Response> {
  const { systemPrompt, tailText, toolResults } = parseMessages(body.messages);
  const modelId = body.model;
  const tools = body.tools ?? [];

  if (!body.messages.some((m) => m.role === "user" || m.role === "tool")) {
    return new Response(
      JSON.stringify({
        error: {
          message: "No user message found",
          type: "invalid_request_error",
        },
      }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  }

  // A request whose trailing tool results answer a paused run resumes it,
  // unless the conversation was rewritten under it (takePausedRun).
  const promptTokens = requestTokens(body.messages, tools);
  const paused = takePausedRun(body.messages);
  if (paused) {
    const resumed = resumeRun(paused, toolResults, tailText, body.stream !== false, promptTokens);
    if (resumed) return resumed;
    console.error(`[bridge] run ${paused.id} ended before its tool results came back; answering statelessly`);
  }

  // helm patch (2026-09-16): TOOL RESULTS WERE SILENTLY DROPPED ON THE
  // STATELESS PATH, so the model saw its own tool call with no output and
  // concluded the tool was broken. Every request that reaches here carries
  // its whole conversation — tool calls and results included — as structured
  // history (toHistory), so nothing the client sent is dropped.
  const { history, userText } = toHistory(body.messages);
  const payload = buildCursorRequest(modelId, systemPrompt, history, userText);
  payload.mcpTools = buildMcpToolDefinitions(tools);
  payload.conversation = conversationStamp(body.messages);

  return startRun(payload, accessToken, modelId, body.stream !== false, promptTokens);
}

// --- Message Parsing ---

interface ToolResultInfo {
  toolCallId: string;
  content: string;
}

interface ParsedMessages {
  systemPrompt: string;
  /** User text after the last assistant message (rides a resumed run's results). */
  tailText: string;
  /** Every tool message in the request, for a paused run's pending calls. */
  toolResults: ToolResultInfo[];
}

/**
 * helm patch (2026-09-16): CONTENT IS A STRING *OR* AN ARRAY OF BLOCKS, AND
 * THIS FILE ASSUMED STRING EVERYWHERE. parseMessages read `msg.content ?? ""`
 * directly, so an array reached the wire as the literal text "[object Object]".
 * Claude Code sends array-form content on essentially every message, so a real
 * seat's whole conversation — system prompt included — arrived at Cursor as
 * that garbage string, and Cursor answered with Connect failed_precondition.
 * MEASURED: a single {type:"text"} block made the model reply "Looks like the
 * message didn't come through correctly ([object O…". Flatten to text here so
 * every caller keeps its string contract.
 */
function contentToText(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const block of content) {
      if (typeof block === "string") { parts.push(block); continue; }
      if (!block || typeof block !== "object") continue;
      const b = block as Record<string, unknown>;
      if (typeof b.text === "string") { parts.push(b.text); continue; }
      // tool_result blocks nest their payload one level deeper.
      if (b.type === "tool_result" && b.content != null) {
        parts.push(contentToText(b.content));
        continue;
      }
      // Anything else (image, tool_use) has no text face; name it rather than
      // dropping it silently, so a lost block is visible instead of invisible.
      if (typeof b.type === "string") parts.push(`[${b.type}]`);
    }
    return parts.join("\n");
  }
  return String(content);
}

function parseMessages(messages: OpenAIMessage[]): ParsedMessages {
  const systemParts = messages
    .filter((m) => m.role === "system")
    .map((m) => contentToText(m.content));
  const toolResults = messages
    .filter((m) => m.role === "tool")
    .map((m) => ({ toolCallId: m.tool_call_id ?? "", content: contentToText(m.content) }));
  const tail: string[] = [];
  for (let i = messages.length - 1; i >= 0 && messages[i]!.role !== "assistant"; i--) {
    if (messages[i]!.role === "user") tail.unshift(contentToText(messages[i]!.content));
  }
  return {
    systemPrompt: systemParts.length > 0 ? systemParts.join("\n") : "You are a helpful assistant.",
    tailText: tail.filter(Boolean).join("\n\n"),
    toolResults,
  };
}

// --- History ---

/**
 * One prior message, in the shape Cursor reads from a root-prompt blob: the
 * AI SDK ModelMessage (role + typed parts), with tool calls and tool results
 * as their own parts. Reference for the shapes: edgebyte-ai/cliproxyapi-
 * cursor-provider-plugin (MIT), translate.go normalizeHistoryMessages.
 */
interface HistoryMessage {
  role: "user" | "assistant" | "tool";
  id?: string;
  content: Array<Record<string, unknown>>;
}

/**
 * The name Cursor gives a caller tool (McpArgs.name, MEASURED: "opencode-Bash"
 * for our "Bash"). History records a past call under it, so the model reads
 * its own past calls as calls to the MCP tool it should use again.
 */
function mcpToolName(name: string): string {
  return `${MCP_PROVIDER}-${name}`;
}

/** What Cursor is asked when the request ends in tool results, not a user message. */
const CONTINUE_PROMPT = "Continue from where you left off, using the tool results above.";

function parseToolArgs(args: string): unknown {
  try {
    return JSON.parse(args || "{}");
  } catch {
    return { _raw: args };
  }
}

/**
 * helm patch (2026-09-25): HISTORY GOES TO CURSOR AS STRUCTURED MESSAGES.
 * The 2026-09-16 workaround folded every prior turn into one plain-text
 * transcript inside the new user message, because synthesised
 * ConversationTurnStructure turns drew failed_precondition. That fold kept
 * user and assistant TEXT only: every tool call vanished from the
 * transcript, and every tool result in the whole conversation was pasted in
 * a block at the top of the latest message, cut off from the call it
 * answered (ds4pro review F2). A Claude Code seat's history is mostly tool
 * calls, so each new user turn showed the model a garbled record of its own
 * work. Each prior message is now its own root-prompt blob, with tool calls
 * and tool results as typed parts in order; the latest user message is the
 * action (or, when the request ends in tool results, a request to continue).
 */
function toHistory(messages: OpenAIMessage[]): { history: HistoryMessage[]; userText: string } {
  const convo = messages.filter((m) => m.role !== "system");
  const last = convo.at(-1);
  const endsWithUser = last?.role === "user";
  const userText = endsWithUser ? contentToText(last!.content) : "";
  const toolNames = new Map<string, string>();
  const history: HistoryMessage[] = [];
  for (const m of endsWithUser ? convo.slice(0, -1) : convo) {
    if (m.role === "user") {
      const text = contentToText(m.content);
      if (text) history.push({ role: "user", content: [{ type: "text", text }] });
    } else if (m.role === "assistant") {
      const parts: Array<Record<string, unknown>> = [];
      const text = contentToText(m.content);
      if (text) parts.push({ type: "text", text });
      for (const call of m.tool_calls ?? []) {
        const name = mcpToolName(call.function.name);
        toolNames.set(call.id, name);
        parts.push({
          type: "tool-call",
          toolCallId: call.id,
          toolName: name,
          args: parseToolArgs(call.function.arguments),
        });
      }
      if (parts.length) history.push({ role: "assistant", content: parts });
    } else if (m.role === "tool") {
      const id = m.tool_call_id ?? "";
      history.push({
        role: "tool",
        id,
        content: [{
          type: "tool-result",
          toolCallId: id,
          toolName: toolNames.get(id) ?? "",
          result: contentToText(m.content),
        }],
      });
    }
  }
  return { history, userText: userText.trim() ? userText : CONTINUE_PROMPT };
}

// --- MCP Tool Definitions ---

/** The MCP provider every caller tool is registered under. */
const MCP_PROVIDER = "opencode";

/** Convert OpenAI tool definitions to Cursor's MCP tool protobuf format. */
function buildMcpToolDefinitions(tools: OpenAIToolDef[]): McpToolDefinition[] {
  return tools.map((t) => {
    const fn = t.function;
    const jsonSchema: JsonValue =
      fn.parameters && typeof fn.parameters === "object"
        ? (fn.parameters as JsonValue)
        : { type: "object", properties: {}, required: [] };
    const inputSchema = toBinary(ValueSchema, fromJson(ValueSchema, jsonSchema));
    return create(McpToolDefinitionSchema, {
      name: fn.name,
      description: fn.description || "",
      providerIdentifier: MCP_PROVIDER,
      toolName: fn.name,
      inputSchema,
    });
  });
}

/** Decode a Cursor MCP arg value (protobuf Value bytes) to a JS value. */
function decodeMcpArgValue(value: Uint8Array): unknown {
  try {
    const parsed = fromBinary(ValueSchema, value);
    return toJson(ValueSchema, parsed);
  } catch {}
  return new TextDecoder().decode(value);
}

/** Decode a map of MCP arg values. */
function decodeMcpArgsMap(args: Record<string, Uint8Array>): Record<string, unknown> {
  const decoded: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    decoded[key] = decodeMcpArgValue(value);
  }
  return decoded;
}

// --- gRPC Request Building ---

function buildCursorRequest(
  modelId: string,
  systemPrompt: string,
  history: HistoryMessage[],
  userText: string,
): CursorRequestPayload {
  // Every root-prompt message is a blob Cursor asks for back by its sha256
  // (KV getBlobArgs); the system prompt leads, then the history in order.
  const blobStore = new Map<string, Uint8Array>();
  const rootPromptMessagesJson: Uint8Array[] = [];
  const addRoot = (message: object): void => {
    const bytes = new TextEncoder().encode(JSON.stringify(message));
    const id = new Uint8Array(createHash("sha256").update(bytes).digest());
    blobStore.set(Buffer.from(id).toString("hex"), bytes);
    rootPromptMessagesJson.push(id);
  };
  addRoot({ role: "system", content: systemPrompt });
  for (const message of history) addRoot(message);

  const conversationState = create(ConversationStateStructureSchema, {
    rootPromptMessagesJson,
    turns: [],
    todos: [],
    pendingToolCalls: [],
    previousWorkspaceUris: [],
    fileStates: {},
    fileStatesV2: {},
    summaryArchives: [],
    turnTimings: [],
    subagentStates: {},
    selfSummaryCount: 0,
    readPaths: [],
  });

  const userMessage = create(UserMessageSchema, {
    text: userText,
    messageId: crypto.randomUUID(),
  });
  const action = create(ConversationActionSchema, {
    action: {
      case: "userMessageAction",
      value: create(UserMessageActionSchema, { userMessage }),
    },
  });

  const modelDetails = create(ModelDetailsSchema, {
    modelId,
    displayModelId: modelId,
    displayName: modelId,
  });

  const runRequest = create(AgentRunRequestSchema, {
    conversationState,
    action,
    modelDetails,
    conversationId: crypto.randomUUID(),
  });

  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "runRequest", value: runRequest },
  });

  return {
    requestBytes: toBinary(AgentClientMessageSchema, clientMessage),
    blobStore,
    mcpTools: [],
  };
}

// --- Connect Protocol Helpers ---

function parseConnectEndStream(data: Uint8Array): Error | null {
  try {
    const payload = JSON.parse(new TextDecoder().decode(data));
    const error = payload?.error;
    if (error) {
      const code = typeof error.code === "string" ? error.code : "unknown";
      const message = typeof error.message === "string" ? error.message : "Unknown error";
      return new Error(`Connect error ${code}: ${message}`);
    }
    return null;
  } catch {
    return new Error("Failed to parse Connect end stream");
  }
}

// helm patch (2026-09-16): CURSOR'S QUOTA REFUSAL MUST BE A 429, NOT TEXT.
// A walled account answers with a Connect end-stream whose error.code is
// resource_exhausted (unavailable has the same shape). Emitting that as
// message content under HTTP 200 made every walled turn look like a finished
// answer: the proxy never cooled the credential, helm never read a wall, and
// the seat respawned reviewers that each died the same way (measured on the
// cursor seat, 2026-09-16). The fresh-request path now PEEKS at the first
// frame before committing to an SSE 200; a refusal there becomes a real 429
// with Retry-After, which CLIProxyAPI cools on and helm surfaces as
// PROXY-COOLDOWN.
//
// helm patch (2026-09-26): THE RESUME PATH WAITS FOR ITS FIRST FRAME TOO, AND
// NO CONNECT ERROR IS EVER REPLY TEXT. MEASURED on the cursor seat: the
// request carrying the owner's tool results resumed a paused run, which
// answered 200 at once, so Cursor's `unavailable` reached Claude Code as the
// reply "[Error: Connect error unavailable: Error]" and the turn ended there.
// A resumed run now answers only after its first frame past heartbeats
// (awaitFirstFrame), and a refusal there is the same 429. A Connect error that
// arrives after the status line is gone ends the stream with a bare JSON
// error line and no finish chunk (see `fail`), which CLIProxyAPI's
// OpenAI-compatible reader turns into an Anthropic `event: error`; the
// refusal codes carry overloaded_error there, a type Claude Code retries.
const CONNECT_REFUSAL_CODES = new Set(["resource_exhausted", "unavailable"]);

function connectRefusal(err: Error): { code: string; message: string } | null {
  const m = /^Connect error ([a-z_]+): ([\s\S]*)$/.exec(err.message);
  if (!m) return null;
  const code = m[1]!;
  return CONNECT_REFUSAL_CODES.has(code) ? { code, message: m[2]! } : null;
}

/** A refusal answered before any byte of the response: 429 with Retry-After. */
function refusalResponse(refusal: { code: string; message: string }): Response {
  return new Response(
    JSON.stringify({
      error: {
        message: `Connect error ${refusal.code}: ${refusal.message}`,
        type: "rate_limit_error",
        code: refusal.code,
      },
    }),
    { status: 429, headers: { "Content-Type": "application/json", "Retry-After": "60" } },
  );
}

/**
 * The error a Connect failure ends a response with once its status line is
 * gone. overloaded_error for the refusal codes: CLIProxyAPI keeps the type
 * when it turns this into an Anthropic `event: error`, and Claude Code retries
 * a streamed error whose body says overloaded_error. Any other code is an
 * api_error: the turn ends as an API error, never as words the model said.
 */
function failureBody(err: Error) {
  const refusal = connectRefusal(err);
  return {
    error: {
      type: refusal ? "overloaded_error" : "api_error",
      code: refusal?.code ?? "connect_error",
      message: err.message,
    },
  };
}

/** A Connect failure answered before the status line: a real HTTP error. */
function failureResponse(err: Error): Response {
  const refusal = connectRefusal(err);
  if (refusal) return refusalResponse(refusal);
  return new Response(JSON.stringify(failureBody(err)), {
    status: 502,
    headers: { "Content-Type": "application/json" },
  });
}

type FirstFramePeek = {
  buf: Buffer;
  refusal: { code: string; message: string } | null;
  closedCode: number | null;
};

// Accumulates bridge output until ONE complete Connect frame is present (or
// the bridge closes, or the timeout passes). Everything received stays in
// `buf` — including bytes that arrive after resolution — so the stream that
// takes over reads `buf` at handover and loses nothing.
function peekFirstFrame(
  bridge: ReturnType<typeof spawnBridge>,
  timeoutMs: number,
): Promise<FirstFramePeek> {
  const peek: FirstFramePeek = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(peek);
    };
    const timer = setTimeout(finish, timeoutMs);
    bridge.onData((chunk) => {
      peek.buf = Buffer.concat([peek.buf, chunk]);
      if (done || peek.buf.length < 5) return;
      const flags = peek.buf[0]!;
      const msgLen = peek.buf.readUInt32BE(1);
      if (peek.buf.length < 5 + msgLen) return;
      if (flags & CONNECT_END_STREAM_FLAG) {
        const err = parseConnectEndStream(peek.buf.subarray(5, 5 + msgLen));
        peek.refusal = err ? connectRefusal(err) : null;
      }
      finish();
    });
    bridge.onClose((code) => {
      peek.closedCode = code;
      finish();
    });
  });
}

function makeHeartbeatBytes(): Uint8Array {
  const heartbeat = create(AgentClientMessageSchema, {
    message: {
      case: "clientHeartbeat",
      value: create(ClientHeartbeatSchema, {}),
    },
  });
  return frameConnectMessage(toBinary(AgentClientMessageSchema, heartbeat));
}

// --- Server Message Processing ---

/**
 * Token counts Cursor reports on the stream: `output` from each
 * InteractionUpdate.token_delta (tokens generated since the last one), and
 * `context` from a checkpoint's ConversationTokenDetails.used_tokens (the
 * conversation's size in the model's window).
 */
interface TokenCounts {
  output?: number;
  context?: number;
}

function processServerMessage(
  msg: AgentServerMessage,
  blobStore: Map<string, Uint8Array>,
  mcpTools: McpToolDefinition[],
  sendFrame: (data: Uint8Array) => void,
  onText: (text: string, isThinking?: boolean) => void,
  onMcpExec: (exec: PendingExec) => void,
  onTokens: (counts: TokenCounts) => void = () => {},
): void {
  const msgCase = msg.message.case;
  if (TRACE) {
    const sub = msg.message.value as
      | { message?: { case?: string }; query?: { case?: string }; $unknown?: never[] }
      | undefined;
    trace("frame", msgCase ?? `unknown(${unknownFields(msg)})`,
      sub?.message?.case ?? sub?.query?.case ?? (sub ? `unknown(${unknownFields(sub)})` : ""));
  }

  if (msgCase === "interactionUpdate") {
    const update = msg.message.value;
    if (update.message.case === "tokenDelta") {
      onTokens({ output: update.message.value.tokens });
    }
    handleInteractionUpdate(update, onText);
  } else if (msgCase === "conversationCheckpointUpdate") {
    const used = msg.message.value.tokenDetails?.usedTokens;
    trace("checkpoint tokenDetails", msg.message.value.tokenDetails
      ? `used=${used} max=${msg.message.value.tokenDetails.maxTokens}` : "absent");
    if (used) onTokens({ context: used });
  } else if (msgCase === "kvServerMessage") {
    handleKvMessage(msg.message.value as KvServerMessage, blobStore, sendFrame);
  } else if (msgCase === "execServerMessage") {
    handleExecMessage(
      msg.message.value as ExecServerMessage,
      mcpTools,
      sendFrame,
      onMcpExec,
    );
  }
}

/**
 * Handle interaction updates — text and thinking only.
 * MCP tool calls are handled entirely via mcpArgs exec messages,
 * not through interaction update callbacks.
 */
function handleInteractionUpdate(
  update: any,
  onText: (text: string, isThinking?: boolean) => void,
): void {
  const updateCase = update.message?.case;

  if (updateCase === "textDelta") {
    const delta = update.message.value.text || "";
    if (delta) onText(delta, false);
  } else if (updateCase === "thinkingDelta") {
    const delta = update.message.value.text || "";
    if (delta) onText(delta, true);
  }
  // toolCallStarted, partialToolCall, toolCallDelta, toolCallCompleted
  // are intentionally ignored. MCP tool calls flow through the exec
  // message path (mcpArgs → mcpResult), not interaction updates.
}

function handleKvMessage(
  kvMsg: KvServerMessage,
  blobStore: Map<string, Uint8Array>,
  sendFrame: (data: Uint8Array) => void,
): void {
  const kvCase = kvMsg.message.case;

  if (kvCase === "getBlobArgs") {
    const blobId = kvMsg.message.value.blobId;
    const blobIdKey = Buffer.from(blobId).toString("hex");
    const blobData = blobStore.get(blobIdKey);

    const response = create(KvClientMessageSchema, {
      id: kvMsg.id,
      message: {
        case: "getBlobResult",
        value: create(GetBlobResultSchema, blobData ? { blobData } : {}),
      },
    });

    const clientMsg = create(AgentClientMessageSchema, {
      message: { case: "kvClientMessage", value: response },
    });
    sendFrame(
      frameConnectMessage(toBinary(AgentClientMessageSchema, clientMsg)),
    );
  } else if (kvCase === "setBlobArgs") {
    const { blobId, blobData } = kvMsg.message.value;
    blobStore.set(Buffer.from(blobId).toString("hex"), blobData);

    const response = create(KvClientMessageSchema, {
      id: kvMsg.id,
      message: {
        case: "setBlobResult",
        value: create(SetBlobResultSchema, {}),
      },
    });

    const clientMsg = create(AgentClientMessageSchema, {
      message: { case: "kvClientMessage", value: response },
    });
    sendFrame(
      frameConnectMessage(toBinary(AgentClientMessageSchema, clientMsg)),
    );
  }
}

/** agent.v1.ExecServerMessage.mcp_state_exec_args / ExecClientMessage.mcp_state_exec_result. */
const MCP_STATE_EXEC_FIELD = 36;

function handleExecMessage(
  execMsg: ExecServerMessage,
  mcpTools: McpToolDefinition[],
  sendFrame: (data: Uint8Array) => void,
  onMcpExec: (exec: PendingExec) => void,
): void {
  const execCase = execMsg.message.case;

  if (execCase === "requestContextArgs") {
    const requestContext = create(RequestContextSchema, {
      rules: [],
      repositoryInfo: [],
      tools: mcpTools,
      gitRepos: [],
      projectLayouts: [],
      mcpInstructions: [],
      fileContents: {},
      customSubagents: [],
    });
    const result = create(RequestContextResultSchema, {
      result: {
        case: "success",
        value: create(RequestContextSuccessSchema, { requestContext }),
      },
    });
    sendExecResult(execMsg, "requestContextResult", result, sendFrame);
    return;
  }

  if (execCase === "mcpArgs") {
    const mcpArgs = execMsg.message.value;
    const decoded = decodeMcpArgsMap(mcpArgs.args ?? {});
    trace(`mcpArgs name=${mcpArgs.name} toolName=${mcpArgs.toolName} provider=${mcpArgs.providerIdentifier}`);
    const cursorToolCallId = mcpArgs.toolCallId || crypto.randomUUID();
    onMcpExec({
      execId: execMsg.execId,
      execMsgId: execMsg.id,
      toolCallId: clientToolCallId(cursorToolCallId),
      cursorToolCallId,
      toolName: mcpArgs.toolName || mcpArgs.name,
      decodedArgs: JSON.stringify(decoded),
    });
    return;
  }

  // helm patch (2026-09-25): CURSOR ASKS FOR MCP STATE BEFORE IT RUNS AN MCP
  // TOOL, AND THE VENDORED PROTO CANNOT READ THE QUESTION. MEASURED on
  // grok-4.7-high: after toolCallStarted the server sends an exec on field 36
  // (mcp_state_exec_args, provider "opencode"), which this proto decodes with
  // no case at all. Nothing answered it, Cursor waited, and the h2 child's
  // 120 s timer killed the stream: the client got the model's "I'll run that
  // with the Bash tool" and then a clean stop with no tool call — a two-minute
  // hang that ends in an empty turn. The field numbers come from Cursor's own
  // CLI bundle (agent.v1.ExecClientMessage.mcp_state_exec_result = 36;
  // McpStateExecResult.success = 1; McpStateSuccess.servers = 1;
  // McpStateServer: server_name = 1, server_identifier = 2, tools = 5).
  // Answer with our one server and its tools, written as an unknown field.
  if (execCase === undefined && execMsg.$unknown?.some((u) => u.no === MCP_STATE_EXEC_FIELD)) {
    const server = new BinaryWriter()
      .tag(1, WireType.LengthDelimited).string(MCP_PROVIDER)
      .tag(2, WireType.LengthDelimited).string(MCP_PROVIDER);
    for (const tool of mcpTools) {
      server.tag(5, WireType.LengthDelimited).bytes(toBinary(McpToolDefinitionSchema, tool));
    }
    const success = new BinaryWriter().tag(1, WireType.LengthDelimited).bytes(server.finish()).finish();
    const result = new BinaryWriter().tag(1, WireType.LengthDelimited).bytes(success).finish();
    const execClientMessage = create(ExecClientMessageSchema, { id: execMsg.id, execId: execMsg.execId });
    execClientMessage.$unknown = [{
      no: MCP_STATE_EXEC_FIELD,
      wireType: WireType.LengthDelimited,
      data: new BinaryWriter().bytes(result).finish(),
    }];
    const clientMessage = create(AgentClientMessageSchema, {
      message: { case: "execClientMessage", value: execClientMessage },
    });
    sendFrame(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
    return;
  }

  // --- Reject native Cursor tools ---
  // The model tries these first. We must respond with rejection/error
  // so it falls back to our MCP tools (registered via RequestContext), and
  // the rejection names the route there (rejectReason).
  const REJECT_REASON = rejectReason(execCase, mcpTools);

  if (execCase === "readArgs") {
    const args = execMsg.message.value;
    const result = create(ReadResultSchema, {
      result: { case: "rejected", value: create(ReadRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "readResult", result, sendFrame);
    return;
  }
  if (execCase === "lsArgs") {
    const args = execMsg.message.value;
    const result = create(LsResultSchema, {
      result: { case: "rejected", value: create(LsRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "lsResult", result, sendFrame);
    return;
  }
  if (execCase === "grepArgs") {
    const result = create(GrepResultSchema, {
      result: { case: "error", value: create(GrepErrorSchema, { error: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "grepResult", result, sendFrame);
    return;
  }
  if (execCase === "writeArgs") {
    const args = execMsg.message.value;
    const result = create(WriteResultSchema, {
      result: { case: "rejected", value: create(WriteRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "writeResult", result, sendFrame);
    return;
  }
  if (execCase === "deleteArgs") {
    const args = execMsg.message.value;
    const result = create(DeleteResultSchema, {
      result: { case: "rejected", value: create(DeleteRejectedSchema, { path: args.path, reason: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "deleteResult", result, sendFrame);
    return;
  }
  if (execCase === "shellArgs" || execCase === "shellStreamArgs") {
    const args = execMsg.message.value;
    const rejected = create(ShellRejectedSchema, {
      command: args.command ?? "",
      workingDirectory: args.workingDirectory ?? "",
      reason: REJECT_REASON,
      isReadonly: false,
    });
    // helm patch (2026-09-25): A STREAMED SHELL IS ANSWERED ON THE STREAM
    // CHANNEL. shell_stream_args expects ExecClientMessage.shell_stream (a
    // ShellStream event); this sent shell_result, which Cursor does not read
    // for that exec, so it waited on an answer that never came. MEASURED on
    // grok-4.7-high: the model reached for Cursor's own shell, the bridge
    // "rejected" it with the wrong message, and the turn sat on heartbeats
    // until the HTTP idle timeout cut it at 254 s ("unexpected EOF").
    if (execCase === "shellStreamArgs") {
      sendExecResult(execMsg, "shellStream",
        create(ShellStreamSchema, { event: { case: "rejected", value: rejected } }), sendFrame);
    } else {
      sendExecResult(execMsg, "shellResult",
        create(ShellResultSchema, { result: { case: "rejected", value: rejected } }), sendFrame);
    }
    return;
  }
  if (execCase === "backgroundShellSpawnArgs") {
    const args = execMsg.message.value;
    const result = create(BackgroundShellSpawnResultSchema, {
      result: {
        case: "rejected",
        value: create(ShellRejectedSchema, {
          command: args.command ?? "",
          workingDirectory: args.workingDirectory ?? "",
          reason: REJECT_REASON,
          isReadonly: false,
        }),
      },
    });
    sendExecResult(execMsg, "backgroundShellSpawnResult", result, sendFrame);
    return;
  }
  if (execCase === "writeShellStdinArgs") {
    const result = create(WriteShellStdinResultSchema, {
      result: { case: "error", value: create(WriteShellStdinErrorSchema, { error: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "writeShellStdinResult", result, sendFrame);
    return;
  }
  if (execCase === "fetchArgs") {
    const args = execMsg.message.value;
    const result = create(FetchResultSchema, {
      result: { case: "error", value: create(FetchErrorSchema, { url: args.url ?? "", error: REJECT_REASON }) },
    });
    sendExecResult(execMsg, "fetchResult", result, sendFrame);
    return;
  }
  if (execCase === "diagnosticsArgs") {
    const result = create(DiagnosticsResultSchema, {});
    sendExecResult(execMsg, "diagnosticsResult", result, sendFrame);
    return;
  }

  // MCP resource/screen/computer exec types
  const miscCaseMap: Record<string, string> = {
    listMcpResourcesExecArgs: "listMcpResourcesExecResult",
    readMcpResourceExecArgs: "readMcpResourceExecResult",
    recordScreenArgs: "recordScreenResult",
    computerUseArgs: "computerUseResult",
  };
  const resultCase = miscCaseMap[execCase as string];
  if (resultCase) {
    sendExecResult(execMsg, resultCase, create(McpResultSchema, {}), sendFrame);
    return;
  }

  // Unknown exec type — log and ignore
  console.error(`[proxy] unhandled exec: ${execCase ?? `unknown(${unknownFields(execMsg)})`}`);
}

/**
 * helm patch (2026-09-26): A REFUSED CURSOR TOOL NAMES THE ROUTE TO THE
 * CALLER'S TOOL. MEASURED on the cursor seat: grok-4.7-high read "Tool not
 * available in this environment. Use the MCP tools provided instead." as the
 * end of the road and ended turns saying every command channel was
 * unavailable, although the same session had reached the caller's Bash
 * through CallDynamicTool before. The caller's tools reach the model as MCP
 * server MCP_PROVIDER, called through Cursor's CallDynamicTool (namespace =
 * the server, toolName = the tool), with GetDynamicTools for the schema. Each
 * refusal names the caller's tool that does the refused tool's job (first
 * of its preferences that the request registered), or the route alone when
 * none is registered.
 */
const NATIVE_TOOL_EQUIVALENT: Record<string, string[]> = {
  shellArgs: ["Bash"],
  shellStreamArgs: ["Bash"],
  backgroundShellSpawnArgs: ["Bash"],
  writeShellStdinArgs: ["Bash"],
  readArgs: ["Read", "Bash"],
  lsArgs: ["Bash", "Glob"],
  grepArgs: ["Grep", "Bash"],
  writeArgs: ["Write", "Bash"],
  deleteArgs: ["Bash"],
  fetchArgs: ["WebFetch", "Bash"],
};

function rejectReason(execCase: string | undefined, mcpTools: McpToolDefinition[]): string {
  const registered = new Set(mcpTools.map((t) => t.toolName || t.name));
  const tool = (NATIVE_TOOL_EQUIVALENT[execCase ?? ""] ?? []).find((name) => registered.has(name));
  const route = tool
    ? `call the MCP tool "${tool}" on server "${MCP_PROVIDER}" with CallDynamicTool (namespace "${MCP_PROVIDER}", toolName "${tool}"); GetDynamicTools gives its schema`
    : `call a tool on MCP server "${MCP_PROVIDER}" with CallDynamicTool (namespace "${MCP_PROVIDER}"); GetDynamicTools lists them with their schemas`;
  return `Cursor's own tool is refused in this environment, so do not stop here: ${route}.`;
}

/**
 * helm patch (2026-09-25): THE TOOL-CALL ID THE CLIENT SEES MUST SURVIVE THE
 * ROUND TRIP. Cursor's ids carry a newline — MEASURED on grok-4.7-high:
 * "call-<uuid>-1\nfc_<uuid>_0". This bridge handed that id out raw;
 * CLIProxyAPI rewrites every character outside [A-Za-z0-9_-] to "_" on the
 * way to Claude Code (util/claude_tool_id.go), Claude Code sends the rewritten
 * id back with the tool result, and the resume path matched by exact equality.
 * Every match missed, every tool result went to Cursor as "Tool result not
 * provided", and the model said so. MEASURED, same request: "The tool
 * execution failed. Retrying now." with MARKER-42 sitting in the result it
 * was never given. Emit ids already in Claude's alphabet, and compare the
 * client's id through the same function, so an id that was rewritten on the
 * way out and one that was not both land on the pending exec.
 */
function clientToolCallId(id: string): string {
  const safe = id.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
  return safe || `call_${crypto.randomUUID().replace(/-/g, "")}`;
}

/** Send an exec client message back to Cursor. */
function sendExecResult(
  execMsg: ExecServerMessage,
  messageCase: string,
  value: unknown,
  sendFrame: (data: Uint8Array) => void,
): void {
  const execClientMessage = create(ExecClientMessageSchema, {
    id: execMsg.id,
    execId: execMsg.execId,
    message: { case: messageCase as any, value: value as any },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "execClientMessage", value: execClientMessage },
  });
  sendFrame(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
}

// --- Runs: one Cursor stream, across the client's tool round trips ---

interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** Both counts are estimates, UTF-8 bytes over four (usageOf). */
  helm_usage: "estimate";
}

/** Where a run's output goes while a client request is attached to it. */
interface Sink {
  text(text: string, thinking: boolean): void;
  toolCall(exec: PendingExec): void;
  /** Ends the client response; `usage` rides the terminal chunk. */
  finish(reason: "stop" | "tool_calls", usage: Usage): void;
  /** Ends the client response with Cursor's Connect error (see `fail`). */
  fail(err: Error): void;
  /** UTF-8 bytes of the text, reasoning and tool calls this response carried. */
  outBytes: number;
  /** Output tokens Cursor reported during this response (traced only). */
  outputTokens: number;
}

/** Start a fresh Cursor run for this request and answer from it. */
async function startRun(
  payload: CursorRequestPayload,
  accessToken: string,
  modelId: string,
  stream: boolean,
  promptTokens: number,
): Promise<Response> {
  const bridge = spawnBridge(accessToken);
  bridge.write(frameConnectMessage(payload.requestBytes));
  const heartbeatTimer = setInterval(() => {
    bridge.write(makeHeartbeatBytes());
  }, 5_000);
  const peeked = await peekFirstFrame(bridge, limits.firstFrameMs);
  if (peeked.refusal) {
    clearInterval(heartbeatTimer);
    bridge.end();
    console.error(`[bridge] upstream refusal on first frame: ${peeked.refusal.code} -> 429`);
    return refusalResponse(peeked.refusal);
  }
  const run: Run = {
    id: ++runSerial,
    modelId,
    bridge,
    heartbeatTimer,
    blobStore: payload.blobStore,
    mcpTools: payload.mcpTools,
    awaiting: [],
    queued: [],
    held: [],
    sink: null,
    pauseAfterChunk: false,
    pending: Buffer.alloc(0),
    closed: false,
    expiry: null,
    stall: null,
    openTools: new Set(),
    lastByteAt: 0,
    lastProgressAt: 0,
    lastSentAt: 0,
    promptTokens,
    contextTokens: 0,
    heldOutputTokens: 0,
    conversation: payload.conversation,
    failure: null,
    opening: null,
  };
  trace(`run ${run.id} started (${modelId})`);
  return respond(run, stream, () => listen(run, peeked));
}

/** Take over the bridge from the first-frame peek: every byte it held, then the rest. */
function listen(run: Run, peeked: FirstFramePeek): void {
  run.bridge.onData((chunk) => feed(run, chunk));
  run.bridge.onClose((code) => onRunClosed(run, code));
  feed(run, peeked.buf);
  if (peeked.closedCode !== null) onRunClosed(run, peeked.closedCode);
}

function feed(run: Run, chunk: Buffer): void {
  let progressed = false;
  run.lastByteAt = performance.now();
  run.pending = Buffer.concat([run.pending, chunk]);
  while (run.pending.length >= 5) {
    const flags = run.pending[0]!;
    const msgLen = run.pending.readUInt32BE(1);
    if (run.pending.length < 5 + msgLen) break;
    const messageBytes = run.pending.subarray(5, 5 + msgLen);
    run.pending = run.pending.subarray(5 + msgLen);

    if (flags & CONNECT_END_STREAM_FLAG) {
      progressed = true;
      const endError = parseConnectEndStream(messageBytes);
      if (endError) {
        console.error(`[bridge] run ${run.id}: ${endError.message}`);
        run.failure ??= endError;
      }
      continue;
    }
    try {
      const msg = fromBinary(AgentServerMessageSchema, messageBytes);
      if (!(msg.message.case === "interactionUpdate" && msg.message.value.message.case === "heartbeat")) {
        progressed = true;
      }
      trackToolCalls(run, msg);
      processServerMessage(
        msg,
        run.blobStore,
        run.mcpTools,
        (data) => run.bridge.write(data),
        (text, isThinking) => emitText(run, text, Boolean(isThinking)),
        (exec) => onExec(run, exec),
        (counts) => countTokens(run, counts),
      );
    } catch (err) {
      // One bad frame must not end the read loop — that loop is the only
      // thing that ever notices the run finishing.
      console.error(`[bridge] run ${run.id}: frame of ${msgLen} bytes failed:`,
        err instanceof Error ? err.message : String(err));
    }
  }
  if (progressed) run.lastProgressAt = performance.now();
  // A resumed run waiting for its first frame: this chunk decides its status.
  if (run.opening) {
    if (progressed) run.opening();
    return;
  }
  if (run.failure && run.sink) {
    fail(run);
    return;
  }
  // Execs that arrive in one chunk are one batch: pause once, after all of
  // them are on the wire, so parallel tool calls reach the client together.
  if (run.pauseAfterChunk) pause(run);
  armStall(run);
}

/**
 * helm patch (2026-09-27): WHETHER CURSOR WAITS ON THE BRIDGE IS CURSOR'S OWN
 * BOOKKEEPING. MEASURED in the bridge's trace of 2026-09-25: each of the 38
 * tool calls opened with toolCallStarted before its exec, and each exec the
 * bridge answered was followed by toolCallCompleted within 100 ms. The one
 * that stuck (a streamed shell refused on the wrong channel) got no
 * toolCallCompleted, and Cursor sent only heartbeats for 252 s. The bridge
 * cannot tell a wrong answer from a right one, so the open call is the sign.
 */
function trackToolCalls(run: Run, msg: AgentServerMessage): void {
  if (msg.message.case !== "interactionUpdate") return;
  const update = msg.message.value.message;
  if (update.case === "toolCallStarted") run.openTools.add(update.value.callId);
  else if (update.case === "toolCallCompleted") run.openTools.delete(update.value.callId);
}

/**
 * End the attached response with Cursor's Connect error, then the run. The
 * client reads an error it can retry, never a finished turn whose last words
 * are the error.
 */
function fail(run: Run): void {
  const sink = run.sink!;
  run.sink = null;
  run.pauseAfterChunk = false;
  armStall(run);
  sink.fail(run.failure!);
  endRun(run, `Cursor ended it: ${run.failure!.message}`);
}

/**
 * When the attached run is stuck, and why: the earliest of
 * - a dead stream: no byte from Cursor, not even a heartbeat, for stallMs;
 * - Cursor waiting on the bridge: a tool call open and only heartbeats for
 *   stallMs (the 09-25 stall);
 * - the client about to time out: no byte of its response for idleGuardMs,
 *   whatever Cursor is doing, since Bun would close the connection next.
 * A model thinking with no tool call open sends heartbeats, and is none of
 * these until the last.
 */
function stallDue(run: Run): { at: number; why: string } {
  const s = (ms: number) => ms / 1000;
  const due = [
    { at: run.lastByteAt + limits.stallMs,
      why: `Cursor sent nothing for ${s(limits.stallMs)} s, not even a heartbeat` },
    { at: run.lastSentAt + limits.idleGuardMs,
      why: `the client got nothing for ${s(limits.idleGuardMs)} s, and its idle timeout is ${IDLE_TIMEOUT_S} s` },
  ];
  if (run.openTools.size) {
    due.push({ at: run.lastProgressAt + limits.stallMs,
      why: `Cursor made no progress for ${s(limits.stallMs)} s with a tool call open (heartbeats only)` });
  }
  return due.reduce((a, b) => (b.at < a.at ? b : a));
}

/** (Re)start the watchdog while a client waits on this run; clear it otherwise. */
function armStall(run: Run): void {
  if (run.stall) clearTimeout(run.stall);
  run.stall = null;
  if (!run.sink || run.closed) return;
  run.stall = setTimeout(() => {
    run.stall = null;
    if (!run.sink || run.closed) return;
    const due = stallDue(run);
    if (due.at > performance.now()) {
      armStall(run);
      return;
    }
    emitText(run, `\n[Error: ${due.why}; the bridge ended this run]`, false);
    endRun(run, `stalled: ${due.why}`);
  }, Math.max(0, stallDue(run).at - performance.now()));
}

function countTokens(run: Run, counts: TokenCounts): void {
  if (counts.context) run.contextTokens = counts.context;
  if (!counts.output) return;
  if (run.sink) run.sink.outputTokens += counts.output;
  else run.heldOutputTokens += counts.output;
}

function emitText(run: Run, text: string, thinking: boolean): void {
  if (run.sink) run.sink.text(text, thinking);
  else run.held.push({ text, thinking });
}

function onExec(run: Run, exec: PendingExec): void {
  trace(`run ${run.id} tool call ${exec.toolName} id=${exec.toolCallId}`);
  if (!run.sink) {
    run.queued.push(exec);
    return;
  }
  run.awaiting.push(exec);
  run.sink.toolCall(exec);
  run.pauseAfterChunk = true;
}

/** End the attached response with tool_calls and wait for the results. */
function pause(run: Run): void {
  run.pauseAfterChunk = false;
  const sink = run.sink;
  run.sink = null;
  armStall(run);
  if (sink) sink.finish("tool_calls", usageOf(run, sink));
  if (run.closed) return;
  for (const exec of run.awaiting) pausedRuns.set(exec.toolCallId, run);
  run.expiry = setTimeout(() => endRun(run, "no tool results came back"), limits.pausedTtlMs);
  trace(`run ${run.id} paused awaiting ${run.awaiting.map((e) => e.toolCallId).join(",")}`);
}

function onRunClosed(run: Run, code: number): void {
  if (run.closed) return;
  run.closed = true;
  clearInterval(run.heartbeatTimer);
  forget(run);
  armStall(run);
  trace(`run ${run.id} closed (code ${code})`);
  if (run.opening) {
    run.opening();
    return;
  }
  if (run.pauseAfterChunk) {
    pause(run);
    return;
  }
  const sink = run.sink;
  run.sink = null;
  if (sink) sink.finish("stop", usageOf(run, sink));
}

/** Drop a run from the paused map and disarm its expiry. */
function forget(run: Run): void {
  for (const exec of run.awaiting) {
    if (pausedRuns.get(exec.toolCallId) === run) pausedRuns.delete(exec.toolCallId);
  }
  if (run.expiry) clearTimeout(run.expiry);
  run.expiry = null;
}

/** Stop a run nobody will come back for. */
function endRun(run: Run, why: string): void {
  forget(run);
  if (run.closed) return;
  console.error(`[bridge] run ${run.id} ended: ${why}`);
  clearInterval(run.heartbeatTimer);
  run.bridge.end();
  run.bridge.proc.kill();
}

/** The stamp of a client conversation (see ConversationStamp). */
function conversationStamp(messages: OpenAIMessage[]): ConversationStamp {
  const first = messages.find((m) => m.role === "user");
  const head = createHash("sha256").update(contentToText(first?.content ?? "")).digest("hex");
  return { head, seen: messages.length };
}

function stampConversation(run: Run, messages: OpenAIMessage[]): void {
  run.conversation = conversationStamp(messages);
}

/** Does this request continue the conversation the run holds? */
function continues(run: Run, messages: OpenAIMessage[]): boolean {
  if (!run.conversation) return true;
  const now = conversationStamp(messages);
  return now.head === run.conversation.head && now.seen > run.conversation.seen;
}

/**
 * The paused run this request may resume, or null. A run found by its tool
 * ids whose conversation the request does not continue (a compaction
 * rewrote it) is ENDED rather than resumed: resuming kept Cursor on the whole
 * pre-compaction conversation, so the compaction bought nothing and every
 * turn went on billing the old window. The request then goes stateless and
 * Cursor gets the compacted history.
 */
function takePausedRun(messages: OpenAIMessage[]): Run | null {
  const run = findPausedRun(messages);
  if (!run || continues(run, messages)) {
    if (run) stampConversation(run, messages);
    return run;
  }
  endRun(run, "the client's conversation was rewritten (a compaction); its history goes to a new run");
  return null;
}

/**
 * The paused run a request answers: one whose tool call is answered by a tool
 * message after the request's last assistant message.
 */
function findPausedRun(messages: OpenAIMessage[]): Run | null {
  for (let i = messages.length - 1; i >= 0 && messages[i]!.role !== "assistant"; i--) {
    const m = messages[i]!;
    if (m.role !== "tool") continue;
    const run = pausedRuns.get(clientToolCallId(m.tool_call_id ?? ""));
    if (run) return run;
  }
  return null;
}

/**
 * Feed the client's tool results to a paused run and answer from it, or
 * return null when the run has ended and the request must go stateless.
 */
function resumeRun(
  run: Run,
  toolResults: ToolResultInfo[],
  userText: string,
  stream: boolean,
  promptTokens: number,
): Response | Promise<Response> | null {
  forget(run);
  if (run.closed) return null;
  run.promptTokens = promptTokens;   // the request now attached is the whole conversation
  run.contextTokens = 0;   // a checkpoint belongs to the response it arrives in
  const answered = run.awaiting.map((exec) => toolResults.find(
    (r) => clientToolCallId(r.toolCallId) === exec.toolCallId,
  ));
  // Text the client sent alongside its tool results (Claude Code appends
  // reminders, and a user who interrupts types here) has nowhere else to go
  // on a paused run: it rides the last result instead of vanishing.
  let last = answered.length - 1;
  while (last >= 0 && !answered[last]) last--;
  if (userText && last >= 0) {
    answered[last] = { ...answered[last]!, content: `${answered[last]!.content}\n\n<user_message>\n${userText}\n</user_message>` };
  } else if (userText) {
    console.error(`[bridge] run ${run.id}: user text with no matching tool result was dropped`);
  }
  for (const [i, exec] of run.awaiting.entries()) {
    const result = answered[i];
    if (!result) console.error(`[bridge] run ${run.id}: no result for tool call ${exec.toolCallId}`);
    const mcpResult = result
      ? create(McpResultSchema, {
          result: {
            case: "success",
            value: create(McpSuccessSchema, {
              content: [
                create(McpToolResultContentItemSchema, {
                  content: {
                    case: "text",
                    value: create(McpTextContentSchema, { text: result.content }),
                  },
                }),
              ],
              isError: false,
            }),
          },
        })
      : create(McpResultSchema, {
          result: {
            case: "error",
            value: create(McpErrorSchema, { error: "Tool result not provided" }),
          },
        });
    const execClientMessage = create(ExecClientMessageSchema, {
      id: exec.execMsgId,
      execId: exec.execId,
      message: { case: "mcpResult", value: mcpResult },
    });
    const clientMessage = create(AgentClientMessageSchema, {
      message: { case: "execClientMessage", value: execClientMessage },
    });
    run.bridge.write(frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
  }
  run.awaiting = [];
  trace(`run ${run.id} resumed`);
  // Something arrived while no client listened: answer from it at once.
  if (run.held.length || run.queued.length || run.failure) return respond(run, stream, () => {});
  return awaitFirstFrame(run, limits.firstFrameMs).then(() => {
    const refusal = run.failure ? connectRefusal(run.failure) : null;
    if (!refusal) return respond(run, stream, () => {});
    console.error(`[bridge] run ${run.id}: upstream refusal on the resumed run's first frame: ${refusal.code} -> 429`);
    endRun(run, "Cursor refused the resumed run");
    return refusalResponse(refusal);
  });
}

/**
 * Resolve once the resumed run's first frame past heartbeats arrives (feed),
 * the run closes (onRunClosed), or the wait runs out. Until then nothing is
 * attached: what arrives is held, and a refusal can still be a 429.
 */
function awaitFirstFrame(run: Run, timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => run.opening?.(), timeoutMs);
    run.opening = () => {
      clearTimeout(timer);
      run.opening = null;
      resolve();
    };
  });
}

/**
 * Attach a new client response to a run: replay what arrived while no one
 * was listening, hand out tool calls Cursor asked for meanwhile, and finish
 * at once if the run already ended.
 */
function attach(run: Run, sink: Sink): void {
  run.sink = sink;
  // A new response starts every watchdog clock: what came before it was
  // another response's, or a pause while the client ran a tool.
  run.lastByteAt = run.lastProgressAt = run.lastSentAt = performance.now();
  sink.outputTokens += run.heldOutputTokens;
  run.heldOutputTokens = 0;
  for (const h of run.held.splice(0)) sink.text(h.text, h.thinking);
  if (run.failure) {
    fail(run);
    return;
  }
  const queued = run.queued.splice(0);
  for (const exec of queued) {
    run.awaiting.push(exec);
    sink.toolCall(exec);
  }
  if (queued.length) pause(run);
  else if (run.closed) {
    run.sink = null;
    sink.finish("stop", usageOf(run, sink));
  }
  armStall(run);
}

/** Bytes per token for both usage estimates (usageOf). */
const BYTES_PER_TOKEN = 4;

const utf8Bytes = (text: string): number => Buffer.byteLength(text, "utf8");

/**
 * The size of a request in tokens, estimated: the UTF-8 bytes of every
 * message's text (system prompt and tool results included), every tool
 * call's name and arguments, and every tool's name, description and
 * parameter schema, over BYTES_PER_TOKEN. It counts the text the client
 * sent, never the JSON the bridge wraps it in for Cursor.
 */
function requestTokens(messages: OpenAIMessage[], tools: OpenAIToolDef[]): number {
  let bytes = 0;
  for (const m of messages) {
    bytes += utf8Bytes(contentToText(m.content));
    for (const call of m.tool_calls ?? []) {
      bytes += utf8Bytes(call.function.name) + utf8Bytes(call.function.arguments ?? "");
    }
  }
  for (const { function: fn } of tools) {
    bytes += utf8Bytes(fn.name) + utf8Bytes(fn.description ?? "") + utf8Bytes(JSON.stringify(fn.parameters ?? {}));
  }
  return Math.ceil(bytes / BYTES_PER_TOKEN);
}

/**
 * The usage a response reports. Claude Code reads it as the size of its
 * conversation: its auto-compact counts input plus output, and helm's
 * autocompact counts the input.
 *
 * helm patch (2026-09-25): every terminal chunk carries usage. A tool-call
 * turn, most turns of a Claude Code seat, used to carry none (task/2640).
 *
 * helm patch (2026-09-27): USAGE DESCRIBES THE REQUEST, NEVER CURSOR'S COUNT
 * (task/3374). The 2026-09-26 patch reported Cursor's checkpoint count
 * (ConversationTokenDetails.used_tokens) and token_delta output. They count
 * Cursor's own conversation, which is not the one the client sent. MEASURED
 * on the cursor seat's transcripts:
 *  - one conversation, two fresh runs 87 minutes apart, 796 bytes of message
 *    text between them: 77,400, then 911,810, which is 3.6 times Cursor's own
 *    max_tokens of 256,000. helm read 474.9% of the 192,000 window. That
 *    build reported five counts above 256,000 in 21 hours, and a compaction
 *    followed each one within four minutes.
 *  - the first response of a fresh conversation: about 50,500 output tokens
 *    (ten conversations, 50,474 to 50,909) for a few hundred visible tokens
 *    in under 20 s. The next checkpoint rose about 47,000 (16,310 to 63,453)
 *    for 3.9 KB of new conversation. While it worked, the seat compacted
 *    every 8 to 19 minutes.
 *
 * So both counts are ESTIMATES, and `helm_usage` says so. `prompt_tokens` is
 * the attached request's size (`requestTokens`, taken again from every
 * request that resumes the run, since each one carries the whole
 * conversation). `completion_tokens` is the UTF-8 bytes of the text,
 * reasoning and tool calls this response carried, over the same four.
 * Cursor's counts reach neither; a trace line shows them beside the estimate.
 */
function usageOf(run: Run, sink: Sink): Usage {
  const output = Math.ceil(sink.outBytes / BYTES_PER_TOKEN);
  trace(`run ${run.id} usage prompt=${run.promptTokens} completion=${output} (estimates);`,
    `Cursor counted context=${run.contextTokens || "none"} output=${sink.outputTokens}`);
  return {
    prompt_tokens: run.promptTokens,
    completion_tokens: output,
    total_tokens: run.promptTokens + output,
    helm_usage: "estimate",
  };
}

function respond(run: Run, stream: boolean, then: () => void): Response | Promise<Response> {
  return stream ? sseResponse(run, then) : jsonResponse(run, then);
}

function completionIdentity() {
  return {
    id: `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 28)}`,
    created: Math.floor(Date.now() / 1000),
  };
}

function sseResponse(run: Run, then: () => void): Response {
  const { id, created } = completionIdentity();
  let mine: Sink | null = null;
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      // helm patch (2026-09-16): A CLIENT DISCONNECT USED TO KILL THE WHOLE
      // BRIDGE PROCESS. The `closed` flag only tracks OUR OWN closes, so when
      // the client hangs up mid-stream — pressing Esc in Claude Code is enough —
      // the runtime closes the controller underneath us, `closed` is still
      // false, and the next enqueue throws ERR_INVALID_STATE from inside a
      // bridge.onClose callback. Nothing catches it there, so bun exits and
      // EVERY seat on this bridge starts getting ECONNREFUSED. MEASURED: the
      // owner interrupted one turn and the bridge died with
      // "Invalid state: Controller is already closed" at the stop chunk.
      // A dead stream is a normal end of life, never a fatal error.
      let closed = false;
      const safeEnqueue = (text: string) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(text));
          run.lastSentAt = performance.now();
        } catch (err) {
          // helm patch (2026-09-16, ds4pro review F5): do NOT swallow silently.
          // A closed controller is the ordinary end of a stream; anything else
          // is a real defect and used to be indistinguishable from it.
          const msg = err instanceof Error ? err.message : String(err);
          if (!/closed|ERR_INVALID_STATE/i.test(msg)) {
            console.error("[bridge] SSE enqueue failed, not a disconnect:", msg);
          }
          closed = true; // stop writing either way; never throw from a callback
        }
      };
      const chunk = (delta: Record<string, unknown>, finishReason: string | null = null) => ({
        id,
        object: "chat.completion.chunk",
        created,
        model: run.modelId,
        choices: [{ index: 0, delta, finish_reason: finishReason }],
      });
      const send = (data: object) => safeEnqueue(`data: ${JSON.stringify(data)}\n\n`);
      let thinking = false;
      let toolIndex = 0;
      const sink: Sink = {
        outBytes: 0,
        outputTokens: 0,
        text(text, isThinking) {
          sink.outBytes += utf8Bytes(text);
          // helm patch (2026-09-16): REASONING IS A SEPARATE DELTA FIELD, NOT
          // FENCED TEXT. `reasoning_content` on the delta is the OpenAI
          // convention CLIProxyAPI maps to an Anthropic thinking block
          // (internal/translator/openai/claude/openai_claude_response.go);
          // literal <think> spans in `content` rendered as blank lines.
          if (isThinking) {
            send(chunk(thinking ? { reasoning_content: text } : { role: "assistant", reasoning_content: text }));
          } else {
            send(chunk({ content: text }));
          }
          thinking = isThinking;
        },
        toolCall(exec) {
          thinking = false;
          sink.outBytes += utf8Bytes(exec.toolName) + utf8Bytes(exec.decodedArgs);
          send(chunk({
            tool_calls: [{
              index: toolIndex++,
              id: exec.toolCallId,
              type: "function",
              function: { name: exec.toolName, arguments: exec.decodedArgs },
            }],
          }));
        },
        finish(reason, usage) {
          send({ ...chunk({}, reason), usage });
          safeEnqueue("data: [DONE]\n\n");
          if (closed) return;
          closed = true;
          try {
            controller.close();
          } catch { /* already closed by the runtime — not an error */ }
        },
        fail(err) {
          // A BARE JSON LINE, NOT `data:`, AND NO FINISH CHUNK OR [DONE].
          // OpenAI's SSE has no error event; CLIProxyAPI's OpenAI-compatible
          // reader treats a line that starts with "{" as a stream error and
          // hands its body to the Claude handler, which writes it as an
          // Anthropic `event: error` with the type it names. A `data:` error
          // chunk would be translated as an empty delta and dropped.
          safeEnqueue(`${JSON.stringify(failureBody(err))}\n\n`);
          if (closed) return;
          closed = true;
          try {
            controller.close();
          } catch { /* already closed by the runtime — not an error */ }
        },
      };
      mine = sink;
      attach(run, sink);
      then();
    },
    cancel() {
      // The client hung up while this response was the run's live one: no
      // request can ever come back for it, so end it rather than let it run.
      if (mine && run.sink === mine) {
        run.sink = null;
        armStall(run);
        endRun(run, "client disconnected");
      }
    },
  });
  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

function jsonResponse(run: Run, then: () => void): Promise<Response> {
  const { id, created } = completionIdentity();
  const { promise, resolve } = Promise.withResolvers<Response>();
  let content = "";
  let reasoning = "";
  const toolCalls: OpenAIToolCall[] = [];
  const sink: Sink = {
    outBytes: 0,
    outputTokens: 0,
    // helm patch (2026-08-11): thinking and answer are two streams; keep them
    // apart. Measured: splicing them gave "I am Cursor G 4.5.rok.I am ...".
    text(text, isThinking) {
      sink.outBytes += utf8Bytes(text);
      if (isThinking) reasoning += text;
      else content += text;
    },
    toolCall(exec) {
      sink.outBytes += utf8Bytes(exec.toolName) + utf8Bytes(exec.decodedArgs);
      toolCalls.push({
        id: exec.toolCallId,
        type: "function",
        function: { name: exec.toolName, arguments: exec.decodedArgs },
      });
    },
    finish(reason, usage) {
      const message: Record<string, unknown> = { role: "assistant", content };
      if (reasoning) message.reasoning_content = reasoning;
      if (toolCalls.length) message.tool_calls = toolCalls;
      resolve(new Response(
        JSON.stringify({
          id,
          object: "chat.completion",
          created,
          model: run.modelId,
          choices: [{ index: 0, message, finish_reason: reason }],
          usage,
        }),
        { headers: { "Content-Type": "application/json" } },
      ));
    },
    // nothing is sent before the answer is whole, so an error keeps its status
    fail(err) {
      resolve(failureResponse(err));
    },
  };
  attach(run, sink);
  then();
  return promise;
}

// helm arm surface (2026-09-16): the first-frame peek and the refusal matcher
// are exported for test/refusal-to-429.test.ts only.
export const __test = { IDLE_TIMEOUT_S, IDLE_MARGIN_MS, stallWindow, requestTokens, handleChatCompletion, peekFirstFrame, connectRefusal, handleExecMessage, clientToolCallId, respond, listen, findPausedRun, takePausedRun, stampConversation, resumeRun, pausedRuns, limits, toHistory, buildCursorRequest, parseMessages };
