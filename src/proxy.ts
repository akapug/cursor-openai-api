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
 * - A native exec (read, shell, grep, ...) whose caller tool is registered →
 *   that tool's call (translateNative), answered as the native (nativeFrames);
 *   a read brings at most READ_BOUND lines and READ_BYTE_BOUND bytes of a file
 *   into the caller's context
 *
 * HTTP/2 transport is delegated to a Node child process (h2-bridge.mjs)
 * because Bun's node:http2 module is broken.
 */
import { create, fromBinary, fromJson, type JsonValue, toBinary, toJson } from "@bufbuild/protobuf";
import { CURSOR_CLIENT_VERSION, getCursorModels } from "./models";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { BinaryReader, BinaryWriter, WireType } from "@bufbuild/protobuf/wire";
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
  ExecClientControlMessageSchema,
  ExecClientMessageSchema,
  ExecClientStreamCloseSchema,
  FetchErrorSchema,
  FetchResultSchema,
  FetchSuccessSchema,
  GetBlobResultSchema,
  GrepContentMatchSchema,
  GrepContentResultSchema,
  GrepCountResultSchema,
  GrepErrorSchema,
  GrepFileCountSchema,
  GrepFileMatchSchema,
  GrepFilesResultSchema,
  GrepResultSchema,
  GrepSuccessSchema,
  GrepUnionResultSchema,
  KvClientMessageSchema,
  LsDirectoryTreeNodeSchema,
  LsDirectoryTreeNode_FileSchema,
  LsErrorSchema,
  LsRejectedSchema,
  LsResultSchema,
  LsSuccessSchema,
  McpErrorSchema,
  McpInstructionsSchema,
  McpResultSchema,
  McpSuccessSchema,
  McpTextContentSchema,
  McpToolDefinitionSchema,
  McpToolNotFoundSchema,
  McpToolResultContentItemSchema,
  ModelDetailsSchema,
  ReadErrorSchema,
  ReadRejectedSchema,
  ReadResultSchema,
  ReadSuccessSchema,
  RequestContextResultSchema,
  RequestContextSchema,
  RequestContextSuccessSchema,
  SetBlobResultSchema,
  ShellFailureSchema,
  ShellRejectedSchema,
  ShellResultSchema,
  ShellSpawnErrorSchema,
  ShellStreamExitSchema,
  ShellStreamSchema,
  ShellStreamStartSchema,
  ShellStreamStdoutSchema,
  ShellSuccessSchema,
  UserMessageActionSchema,
  UserMessageSchema,
  WriteErrorSchema,
  WriteRejectedSchema,
  WriteResultSchema,
  WriteShellStdinErrorSchema,
  WriteShellStdinResultSchema,
  WriteSuccessSchema,
  type AgentServerMessage,
  type ConversationStateStructure,
  type ExecServerMessage,
  type FetchArgs,
  type GrepArgs,
  type GrepContentMatch,
  type GrepResult,
  type GrepUnionResult,
  type KvServerMessage,
  type LsArgs,
  type McpToolDefinition,
  type ReadArgs,
  type ShellArgs,
  type ShellStream,
  type WriteArgs,
} from "./proto/agent_pb";
import { createHash } from "node:crypto";
import { closeSync, openSync, readSync, statSync } from "node:fs";
import { isAbsolute, join as pathJoin, resolve as pathResolve } from "node:path";

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
  /** Cursor's conversation id on the request (a new one for a replay). */
  conversationId: string;
  /** The request's messageChain, for keeping the conversation when its run closes. */
  chain?: string[];
  /**
   * Set when the request continues a kept Cursor conversation: the entry it
   * was taken from, and the replay of the whole history to run instead when
   * Cursor will not continue it (startRun).
   */
  continued?: { kept: KeptConversation; replay: () => CursorRequestPayload };
}

/**
 * helm patch (2026-09-30, task/3817): A CURSOR CONVERSATION KEPT BETWEEN THE
 * CLIENT'S TURNS. Cursor's own CLI keeps one conversation per chat: after
 * each step Cursor sends a checkpoint (conversationCheckpointUpdate), the CLI
 * stores it with the blobs Cursor wrote (KV setBlobArgs), and the next turn
 * sends that checkpoint back under the same conversation id with only the new
 * user message. This bridge used to throw the checkpoint away and replay the
 * whole client history into a new conversation (random id, every message a
 * root-prompt blob) at the start of every run. MEASURED on the cursor seat's
 * bridge.log: on that cold replay Cursor's count for the same ~140k history
 * swung 107k/214k/426k/905k, over its 256k window in 16 of 24 run starts,
 * and 13 of 15 turns over 256k made no tool call; mid-run it counted ~0.77x
 * of the estimate. A run that closes cleanly now leaves its conversation
 * here, and a request whose history extends it by the reply and new user
 * text continues it (continuation).
 */
interface KeptConversation {
  /** Cursor's conversation id, the same on every run that continues it. */
  id: string;
  /** Cursor's latest checkpoint: the state the next run sends back. */
  state: ConversationStateStructure;
  /** Every blob Cursor may ask back for: the replayed root prompt and what Cursor stored. */
  blobs: Map<string, Uint8Array>;
  /** messageChain of the request the last run answered. */
  chain: string[];
  /** Runs this conversation has carried; 1 is the replay that opened it. */
  runs: number;
  /** When it was kept (Date.now), for eviction. */
  at: number;
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
  /**
   * helm patch (2026-09-28): the Cursor native exec this call stands for
   * (translateNative). Its answer is that native's own result (nativeFrames),
   * never an mcpResult.
   */
  native?: NativeCall;
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
  /** Cursor's conversation id for this run (see KeptConversation). */
  conversationId?: string;
  /** Runs that conversation has carried, this one included. */
  conversationRuns?: number;
  /** The latest checkpoint Cursor sent on this run. */
  checkpoint?: ConversationStateStructure | null;
  /** messageChain of the request now attached (or last attached). */
  chain?: string[];
  /** Why the bridge ended this run itself (endRun); a run so ended is never kept. */
  ended?: string | null;
  /** Bytes this run sent Cursor: the run request, then every answer (blobs, exec results). */
  sent?: { request: number; answers: number; blobs: number; blobBytes: number };
}

/** Paused runs by the client-facing id of every tool call they await. */
const pausedRuns = new Map<string, Run>();
let runSerial = 0;

/** Kept Cursor conversations, by the last digest of the chain they answered. */
const keptConversations = new Map<string, KeptConversation>();

/** The seam tests replace to run a scripted Cursor stream (startRun). */
const seams = { spawn: (accessToken: string): Bridge => spawnBridge(accessToken) };

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
  /**
   * helm patch (2026-09-29): how far into a file the bridge reads to fit a
   * native read to READ_BYTE_BOUND (fitToBytes). A window past it cannot be
   * proven to fit, so that read fails closed.
   */
  readScanBytes: 64 * 1024 * 1024,
  /**
   * helm patch (2026-09-30, task/3817): kept Cursor conversations. A seat and
   * the subagents it spawns are one conversation each; a subagent's is never
   * continued once it returns, so the oldest go past this many.
   */
  keptMax: 16,
  /** A kept conversation nobody continued for this long is dropped. */
  keptTtlMs: 12 * 60 * 60_000,
  /** A conversation whose blobs pass this many bytes is not kept (its next turn replays). */
  keptBlobBytes: 64 * 1024 * 1024,
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

  const chain = messageChain(body.messages);

  // A request whose trailing tool results answer a paused run resumes it,
  // unless the conversation was rewritten under it (takePausedRun). That path
  // sends results directly and synthesizes neither stateless prompt.
  const paused = takePausedRun(body.messages);
  if (paused) {
    paused.chain = chain;
    const promptTokens = requestTokens(body.messages, tools, false);
    const resumed = resumeRun(paused, toolResults, tailText, body.stream !== false, promptTokens);
    if (resumed) return resumed;
    console.error(`[bridge] run ${paused.id} ended before its tool results came back; answering statelessly`);
  }

  const promptTokens = requestTokens(body.messages, tools, true);
  const mcpTools = buildMcpToolDefinitions(tools);

  // helm patch (2026-09-16): TOOL RESULTS WERE SILENTLY DROPPED ON THE
  // STATELESS PATH, so the model saw its own tool call with no output and
  // concluded the tool was broken. Every request that reaches here carries
  // its whole conversation — tool calls and results included — as structured
  // history (toHistory), so nothing the client sent is dropped.
  const replay = (): CursorRequestPayload => {
    const { history, userText } = toHistory(body.messages);
    const payload = buildCursorRequest(modelId, systemPrompt, history, userText);
    payload.mcpTools = mcpTools;
    payload.conversation = conversationStamp(body.messages);
    payload.chain = chain;
    return payload;
  };

  // helm patch (2026-09-30, task/3817): a request that extends a kept Cursor
  // conversation by its reply and new user text sends Cursor only that text,
  // with the conversation's checkpoint; any other request replays its
  // history, and the log says why.
  const found = continuation(body.messages, chain);
  if ("why" in found) {
    const payload = replay();
    console.error(`[bridge] new Cursor conversation ${short(payload.conversationId)}: replaying ${body.messages.length} messages (${kb(payload.blobStore)}); ${found.why}`);
    return startRun(payload, accessToken, modelId, body.stream !== false, promptTokens);
  }
  const payload = buildContinuation(modelId, found.kept, found.text);
  payload.mcpTools = mcpTools;
  payload.conversation = conversationStamp(body.messages);
  payload.chain = chain;
  payload.continued = { kept: found.kept, replay };
  return startRun(payload, accessToken, modelId, body.stream !== false, promptTokens);
}

// --- Kept Cursor conversations (helm patch 2026-09-30, task/3817) ---

/**
 * One digest per prefix of the client's messages: chain[i] covers messages
 * 0..i-1, so chain[0] is the empty conversation. Each message is read the way
 * the bridge reads it (role, text, tool-call id, tool calls), so the fields
 * Cursor never sees (cache markers, thinking signatures) do not break a match.
 */
function messageChain(messages: OpenAIMessage[]): string[] {
  const chain = [createHash("sha256").update("cursor-openai-api conversation").digest("hex")];
  for (const m of messages) {
    const face = JSON.stringify([
      m.role,
      contentToText(m.content),
      m.tool_call_id ?? "",
      (m.tool_calls ?? []).map((c) => [c.id, c.function.name, c.function.arguments ?? ""]),
    ]);
    chain.push(createHash("sha256").update(chain.at(-1)!).update(face).digest("hex"));
  }
  return chain;
}

/** The first eight characters of a conversation id, for log lines. */
const short = (id: string): string => id.slice(0, 8);

/** The bytes a blob store holds, for log lines. */
function blobBytes(blobs: Map<string, Uint8Array>): number {
  let n = 0;
  for (const b of blobs.values()) n += b.length;
  return n;
}

const kb = (blobs: Map<string, Uint8Array>): string => `${(blobBytes(blobs) / 1024).toFixed(1)} KB`;

/** Drop kept conversations past their idle limit, then the oldest past the count limit. */
function pruneKept(now = Date.now()): void {
  for (const [key, kept] of keptConversations) {
    if (now - kept.at > limits.keptTtlMs) keptConversations.delete(key);
  }
  const byAge = [...keptConversations.entries()].sort((a, b) => a[1].at - b[1].at);
  while (byAge.length > limits.keptMax) keptConversations.delete(byAge.shift()![0]);
}

/**
 * The kept conversation this request continues and the new text it adds, or
 * why it continues none. It continues one when its first n messages are
 * exactly the request that conversation's last run answered, message n is
 * the reply that run gave (an assistant message with no tool call), and every
 * message after it is user text. Anything else replays: a compaction, a
 * /clear or an edited history no longer extends the conversation, and a
 * tool result or an assistant turn after the reply is a turn Cursor's
 * conversation never saw.
 */
function continuation(
  messages: OpenAIMessage[],
  chain: string[],
): { kept: KeptConversation; text: string } | { why: string } {
  pruneKept();
  for (let n = messages.length - 2; n >= 1; n--) {
    const kept = keptConversations.get(chain[n]!);
    if (!kept) continue;
    const reply = messages[n]!;
    const added = messages.slice(n + 1);
    const at = `Cursor conversation ${short(kept.id)} covers this history's first ${n} messages`;
    if (reply.role !== "assistant" || reply.tool_calls?.length) {
      return { why: `${at}, but message ${n + 1} is not its reply (${reply.role}${reply.tool_calls?.length ? " with tool calls" : ""})` };
    }
    const other = added.find((m) => m.role !== "user");
    if (other) {
      return { why: `${at}, but a ${other.role} message follows its reply, a turn that conversation never saw` };
    }
    const text = added.map((m) => contentToText(m.content)).filter((t) => t.trim()).join("\n\n");
    if (!text) return { why: `${at}, but the ${added.length} message(s) after its reply carry no text` };
    keptConversations.delete(chain[n]!);
    return { kept, text };
  }
  return { why: divergence(chain) };
}

/** Why no kept conversation is continued: the closest one, and where this history leaves it. */
function divergence(chain: string[]): string {
  if (!keptConversations.size) {
    return "no Cursor conversation is kept (the first request since the bridge started, or the last run did not close cleanly)";
  }
  let best: { kept: KeptConversation; shared: number } | null = null;
  for (const kept of keptConversations.values()) {
    let shared = 0;
    while (shared + 1 < Math.min(kept.chain.length, chain.length) && kept.chain[shared + 1] === chain[shared + 1]) shared++;
    if (!best || shared > best.shared) best = { kept, shared };
  }
  if (!best!.shared) {
    return `no kept Cursor conversation opens the way this history does (a new session or subagent, or a /clear); ${keptConversations.size} kept`;
  }
  const had = best!.kept.chain.length - 1;
  return `the history no longer extends Cursor conversation ${short(best!.kept.id)}: it keeps ${best!.shared} of that conversation's ${had} messages and then differs (a compaction, a rewind or an edited history)`;
}

/**
 * Keep a run's Cursor conversation for the client's next turn: only a run
 * that closed on its own, with no error, no tool call waiting, and a
 * checkpoint to send back. The reply it closed on is the message the next
 * request must carry at chain.length - 1 (continuation).
 */
function keepConversation(run: Run): void {
  const why = run.ended ? `the bridge ended it (${run.ended})`
    : run.failure ? `Cursor ended it with ${run.failure.message}`
    : run.pauseAfterChunk || run.awaiting.length || run.queued.length ? "a tool call is still waiting"
    : !run.checkpoint ? "Cursor sent no checkpoint"
    : !run.chain || !run.conversationId ? "it has no client conversation"
    : "";
  if (why) {
    if (run.conversationId) console.error(`[bridge] run ${run.id}: Cursor conversation ${short(run.conversationId)} is not kept, since ${why}; the next request replays its history`);
    return;
  }
  const bytes = blobBytes(run.blobStore);
  if (bytes > limits.keptBlobBytes) {
    console.error(`[bridge] run ${run.id}: Cursor conversation ${short(run.conversationId!)} is not kept: its blobs hold ${bytes} bytes, over ${limits.keptBlobBytes}; the next request replays its history`);
    return;
  }
  keptConversations.set(run.chain!.at(-1)!, {
    id: run.conversationId!,
    state: run.checkpoint!,
    blobs: run.blobStore,
    chain: run.chain!,
    runs: run.conversationRuns ?? 1,
    at: Date.now(),
  });
  pruneKept();
  trace(`run ${run.id} kept Cursor conversation ${short(run.conversationId!)} (${run.chain!.length - 1} messages, ${bytes} blob bytes)`);
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

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

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
    systemPrompt: systemParts.length > 0 ? systemParts.join("\n") : DEFAULT_SYSTEM_PROMPT,
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

const EMPTY_TOOL_SCHEMA: JsonValue = { type: "object", properties: {}, required: [] };

/** Convert OpenAI tool definitions to Cursor's MCP tool protobuf format. */
function buildMcpToolDefinitions(tools: OpenAIToolDef[]): McpToolDefinition[] {
  return tools.map((t) => {
    const fn = t.function;
    const jsonSchema: JsonValue =
      fn.parameters && typeof fn.parameters === "object"
        ? (fn.parameters as JsonValue)
        : EMPTY_TOOL_SCHEMA;
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

  return runRequestPayload(modelId, conversationState, userText, crypto.randomUUID(), blobStore);
}

/**
 * helm patch (2026-09-30, task/3817): continue a kept Cursor conversation.
 * The run request carries Cursor's own last checkpoint (fields this proto
 * cannot read ride along as they came) under the conversation's id, and only
 * the new user text as its action; Cursor asks for any blob it needs back
 * from the kept store, as Cursor's own CLI serves them from its local one.
 */
function buildContinuation(
  modelId: string,
  kept: KeptConversation,
  userText: string,
): CursorRequestPayload {
  return runRequestPayload(modelId, kept.state, userText, kept.id, kept.blobs);
}

function runRequestPayload(
  modelId: string,
  conversationState: ConversationStateStructure,
  userText: string,
  conversationId: string,
  blobStore: Map<string, Uint8Array>,
): CursorRequestPayload {
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
    conversationId,
  });

  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "runRequest", value: runRequest },
  });

  return {
    requestBytes: toBinary(AgentClientMessageSchema, clientMessage),
    blobStore,
    mcpTools: [],
    conversationId,
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
  /** Any Connect error the first frame ended the stream with, a refusal or not. */
  error?: Error | null;
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
        peek.error = err;
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
  onCheckpoint: (state: ConversationStateStructure) => void = () => {},
  onBlobServed: (bytes: number) => void = () => {},
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
    onCheckpoint(msg.message.value);
  } else if (msgCase === "kvServerMessage") {
    handleKvMessage(msg.message.value as KvServerMessage, blobStore, sendFrame, onBlobServed);
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
  onBlobServed: (bytes: number) => void = () => {},
): void {
  const kvCase = kvMsg.message.case;

  if (kvCase === "getBlobArgs") {
    const blobId = kvMsg.message.value.blobId;
    const blobIdKey = Buffer.from(blobId).toString("hex");
    const blobData = blobStore.get(blobIdKey);
    if (!blobData) console.error(`[bridge] Cursor asked for blob ${blobIdKey.slice(0, 16)} and the store does not hold it`);
    onBlobServed(blobData?.length ?? 0);

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
      mcpInstructions: mcpInstructionsFor(mcpTools),
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

  // helm patch (2026-09-28): A CURSOR TOOL THE CALLER CAN RUN IS RUN AS THE
  // CALLER'S TOOL, NEVER REFUSED. MEASURED with CURSOR_BRIDGE_TRACE=1:
  // grok-4.7-high opens a turn with Cursor's own read. In a short
  // conversation it followed the refusal's prose (readArgs, refused, the
  // field-36 MCP-state exec, mcpArgs opencode-Read). In a Claude Code seat's
  // ~100k-token conversation it did not: it wrote text and ended the turn with
  // no tool call. Once "the tool runtime refused every call" was in its
  // history, every later turn gave up the same way for 30 hours, and the seat
  // looked idle until its context was cleared. A refusal works only while the
  // model obeys prose, and one miss poisons the session. So a native whose
  // caller tool is registered goes to the client as that tool's call, the way
  // an mcpArgs call does, and the client's output goes back to Cursor as the
  // native's own result (nativeFrames). The model never reads a refusal. A
  // request the caller's tool cannot fulfil (a read from the end, a line past
  // the byte bound) is answered by the bridge with that request's error,
  // naming the route to what it asked (BridgeAnswer).
  const native = translateNative(execMsg, mcpTools);
  if (Array.isArray(native)) {
    for (const frame of native) sendFrame(frame);
    return;
  }
  if (native) {
    onMcpExec(native);
    return;
  }

  // --- Reject native Cursor tools ---
  // The model tries these first. We must respond with rejection/error
  // so it falls back to our MCP tools (registered via RequestContext), and
  // the rejection names the route there (rejectReason). Only natives with no
  // honest caller equivalent (NATIVE_ROUTES) get here.
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

/**
 * helm patch (2026-09-30, task/3817): THE CALLER'S TOOLS ARE NAMED IN CURSOR'S
 * OWN MCP INSTRUCTIONS, SO THE MODEL KNOWS THEM WITHOUT GetDynamicTools.
 * MEASURED on the cursor seat (transcript 46c53e63, 16:34-16:35Z): after a
 * compaction the model listed Cursor's own tools (Shell, Grep, StrReplace,
 * Write, Read, Glob, AskQuestion, GetDynamicTools, CallDynamicTool), decided
 * "the Monitor tool is unavailable in this Cursor session" and left its
 * beacon dead, although it had armed Monitor through CallDynamicTool five
 * times that session and never called GetDynamicTools to look. The only other
 * place the route was written was the client's system prompt, which reaches
 * Cursor as one more root-prompt message. RequestContext.mcp_instructions is
 * where Cursor's own CLI puts an MCP server's instructions, and Cursor asks
 * for the request context on every run, so every run, a replayed or a
 * continued one, names each registered tool and the exact call. The list is
 * built from the tools the request registered, so it cannot drift from them.
 */
function mcpInstructionsFor(mcpTools: McpToolDefinition[]) {
  if (!mcpTools.length) return [];
  const lines = mcpTools.map((t) => {
    const name = t.toolName || t.name;
    const first = (t.description || "").replace(/\s+/g, " ").trim().split(/(?<=[.!?])\s/)[0] ?? "";
    return `- ${name}${first ? `: ${first.length > 160 ? `${first.slice(0, 157)}...` : first}` : ""}`;
  });
  const example = mcpTools.find((t) => (t.toolName || t.name) === "Monitor") ?? mcpTools[0]!;
  const instructions = [
    `Every tool of this session is a tool on this MCP server, "${MCP_PROVIDER}". Call one with CallDynamicTool: namespace "${MCP_PROVIDER}", toolName the tool's name below, and the tool's own arguments. For example ${example.toolName || example.name} is CallDynamicTool with namespace "${MCP_PROVIDER}" and toolName "${example.toolName || example.name}".`,
    `These tools exist whether or not you have called GetDynamicTools; call GetDynamicTools only to read a tool's full argument schema. Never conclude that one of them is unavailable without calling it.`,
    `The tools on "${MCP_PROVIDER}":`,
    ...lines,
  ].join("\n");
  return [create(McpInstructionsSchema, { serverName: MCP_PROVIDER, instructions })];
}

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
  sendFrame(execClientFrame(execMsg, messageCase, value));
}

/** One exec client message answering the exec `target` names, framed. */
function execClientFrame(
  target: { id: number; execId: string },
  messageCase: string,
  value: unknown,
): Buffer {
  const execClientMessage = create(ExecClientMessageSchema, {
    id: target.id,
    execId: target.execId,
    message: { case: messageCase as any, value: value as any },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "execClientMessage", value: execClientMessage },
  });
  return frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage));
}

// --- Cursor's own tools, run as the caller's (helm patch 2026-09-28) ---

type NativeCase = "readArgs" | "shellArgs" | "shellStreamArgs" | "grepArgs" | "writeArgs" | "lsArgs" | "fetchArgs";
type NativeArgs = ReadArgs | ShellArgs | GrepArgs | WriteArgs | LsArgs | FetchArgs;
type ExecArgs<C extends NativeCase> = Extract<ExecServerMessage["message"], { case: C }>["value"];

/** A native exec a caller tool call stands for (PendingExec.native). */
interface NativeCall {
  case: NativeCase;
  args: NativeArgs;
}

/** One ExecClientMessage answer: its oneof case and value. */
interface ExecAnswer {
  case: string;
  value: unknown;
}

/** A caller tool's arguments: `need` must fit its schema, `may` goes out where the schema declares it (fitSchema). */
interface CallerArgs {
  need: Record<string, unknown>;
  may?: Record<string, unknown>;
}

interface NativeRoute<A> {
  /** The caller tool that does this native's job. */
  tool: string;
  /** Its answer has a text field, so text sent with the results can ride it (resumeRun). */
  carriesText: boolean;
  /** Answered on a stream: its events, then streamClose, as Cursor's own CLI does. */
  streamed?: boolean;
  /**
   * The caller tool's arguments; the native's answer, given by the bridge
   * with no call, for a request the caller tool cannot fulfil (BridgeAnswer);
   * or null when this call has no honest translation.
   */
  call(a: A): CallerArgs | BridgeAnswer | null;
  /** The native's own result for the client's output (undefined: none came back) to the call `sent`. */
  answer(a: A, out: string | undefined, note: string, sent: Record<string, unknown>): ExecAnswer[];
}

/**
 * A native the bridge answers itself, with no client call: an error result
 * for that one request, naming the route to what it asked. Narrower than a
 * refusal: it says what this request cannot do here, not that the tool is
 * refused.
 */
interface BridgeAnswer {
  answer: ExecAnswer[];
}

const NOT_PROVIDED = "Tool result not provided";

/**
 * What the client's tool did, read from its words: CLIProxyAPI drops a tool
 * result's is_error on the way to OpenAI form, so the text is all the bridge
 * gets. Claude Code wraps a call it refused or could not run in
 * <tool_use_error>; that, or no result at all, is a failure, and anything
 * else is the tool's output.
 */
function outcome(out: string | undefined): { ok: true; text: string } | { ok: false; why: string } {
  if (out === undefined) return { ok: false, why: NOT_PROVIDED };
  const m = /^\s*<tool_use_error>([\s\S]*)<\/tool_use_error>\s*$/.exec(out);
  return m ? { ok: false, why: m[1]! } : { ok: true, text: out };
}

/** A command's exit code and output: Claude Code's Bash starts a non-zero exit with "Exit code N". */
function exitStatus(out: string): { code: number; text: string } {
  const m = /^Exit code (\d+)\n?/.exec(out);
  return m ? { code: Number(m[1]), text: out.slice(m[0].length) } : { code: 0, text: out };
}

/** One single-quoted shell word. */
function shellWord(s: string): string {
  return `'${s.replaceAll("'", `'\\''`)}'`;
}

function lineCount(text: string): number {
  return text ? text.split("\n").length - (text.endsWith("\n") ? 1 : 0) : 0;
}

/** The command a native shell runs as: in its working directory, when it names one. */
function inDirectory(a: ShellArgs): CallerArgs | null {
  // A background shell is one Cursor addresses later by shell id and pid;
  // the caller's Bash gives neither, so it keeps the refusal.
  if (a.isBackground) return null;
  return { need: { command: a.workingDirectory ? `cd ${shellWord(a.workingDirectory)} && ${a.command}` : a.command } };
}

function shellAnswer(a: ShellArgs, out: string | undefined, note: string): ExecAnswer[] {
  const where = { command: a.command, workingDirectory: a.workingDirectory };
  const o = outcome(out);
  if (!o.ok) {
    const result = { case: "spawnError" as const, value: create(ShellSpawnErrorSchema, { ...where, error: o.why + note }) };
    return [{ case: "shellResult", value: create(ShellResultSchema, { result }) }];
  }
  const { code, text } = exitStatus(o.text);
  const result = code === 0
    ? { case: "success" as const, value: create(ShellSuccessSchema, { ...where, exitCode: 0, stdout: text + note }) }
    : { case: "failure" as const, value: create(ShellFailureSchema, { ...where, exitCode: code, stdout: text + note }) };
  return [{ case: "shellResult", value: create(ShellResultSchema, { result }) }];
}

/** A streamed shell's events, in the order Cursor's CLI sends them: start, output, exit. */
function shellStreamAnswer(a: ShellArgs, out: string | undefined, note: string): ExecAnswer[] {
  const event = (e: ShellStream["event"]): ExecAnswer => ({ case: "shellStream", value: create(ShellStreamSchema, { event: e }) });
  const o = outcome(out);
  if (!o.ok) {
    return [event({ case: "rejected", value: create(ShellRejectedSchema, {
      command: a.command, workingDirectory: a.workingDirectory, reason: o.why + note, isReadonly: false,
    }) })];
  }
  const { code, text } = exitStatus(o.text);
  const stdout = text + note;
  return [
    event({ case: "start", value: create(ShellStreamStartSchema, {}) }),
    ...(stdout ? [event({ case: "stdout", value: create(ShellStreamStdoutSchema, { data: stdout }) })] : []),
    event({ case: "exit", value: create(ShellStreamExitSchema, { code, cwd: a.workingDirectory }) }),
  ];
}

/** Cursor's grep reads output_mode "content" when none is given (its CLI's grep executor). */
const grepMode = (a: GrepArgs): string => a.outputMode || "content";

/** One line of Claude Code's content-mode Grep: "path:N:text" (a match) or "path-N-text" (context). */
function grepLine(line: string, files: string[], single: string | null) {
  if (single !== null) {
    const m = /^(\d+)([:-])([\s\S]*)$/.exec(line);
    return m ? { file: single, n: Number(m[1]), text: m[3]!, context: m[2] === "-" } : null;
  }
  // Shortest known file first: a context line whose text holds ":N:" also
  // reads as a match line of a longer, made-up path.
  for (const file of files) {
    const m = line.startsWith(file) ? /^([:-])(\d+)\1([\s\S]*)$/.exec(line.slice(file.length)) : null;
    if (m) return { file, n: Number(m[2]), text: m[3]!, context: m[1] === "-" };
  }
  return null;
}

/**
 * Claude Code's Grep answer read back into Cursor's grep result, one reader
 * per output mode. Every line must land in the result: a line that does not
 * read makes the whole answer the grep's error carrying the text (grepArgs),
 * never a match list with lines missing.
 */
const GREP_READERS: Record<string, (text: string, a: GrepArgs) => GrepUnionResult["result"] | null> = {
  files_with_matches(text) {
    if (text.trim() === "No files found") return { case: "files", value: create(GrepFilesResultSchema, {}) };
    const [head, ...files] = text.split("\n").filter(Boolean);
    if (!head || !/^Found \d+ files?\b/.test(head)) return null;
    return { case: "files", value: create(GrepFilesResultSchema, { files, totalFiles: files.length }) };
  },
  count(text) {
    if (text.trim() === "No matches found") return { case: "count", value: create(GrepCountResultSchema, {}) };
    const counts = text.split("\n")
      .filter((l) => l && !/^Found \d+ total occurrences? across \d+ files?\b/.test(l))
      .map((l) => ({ file: l.slice(0, l.lastIndexOf(":")), count: Number(l.slice(l.lastIndexOf(":") + 1)) }));
    if (counts.some((c) => !c.file || !Number.isInteger(c.count))) return null;
    return { case: "count", value: create(GrepCountResultSchema, {
      counts: counts.map((c) => create(GrepFileCountSchema, c)),
      totalFiles: counts.length,
      totalMatches: counts.reduce((n, c) => n + c.count, 0),
    }) };
  },
  content(text, a) {
    const lines = text.trim() === "No matches found" ? [] : text.split("\n")
      .filter((l) => l && l !== "--" && !/^\[Showing results with pagination = .*\]$/.test(l));
    // One file searched prints "N:text"; a directory prints "path:N:text".
    const single = lines.every((l) => /^\d+[:-]/.test(l)) ? a.path ?? "" : null;
    const files = [...new Set(lines.flatMap((l) => /^(.*?):\d+:/.exec(l)?.[1] ?? []))].sort((x, y) => x.length - y.length);
    const hits = lines.map((l) => grepLine(l, files, single));
    if (hits.some((h) => h === null)) return null;
    const byFile = new Map<string, GrepContentMatch[]>();
    for (const h of hits as NonNullable<(typeof hits)[number]>[]) {
      if (!byFile.has(h.file)) byFile.set(h.file, []);
      byFile.get(h.file)!.push(create(GrepContentMatchSchema, { lineNumber: h.n, content: h.text, isContextLine: h.context }));
    }
    return { case: "content", value: create(GrepContentResultSchema, {
      matches: [...byFile].map(([file, matches]) => create(GrepFileMatchSchema, { file, matches })),
      totalLines: hits.length,
      totalMatchedLines: hits.filter((h) => !h!.context).length,
    }) };
  },
};

/** The line a translated listing ends with: a listing the client cut short or reworded lacks it. */
const LS_END = "-- end of listing --";

/**
 * helm patch (2026-09-29): THE MOST OF A FILE ONE OF CURSOR'S READS BRINGS
 * INTO THE CALLER'S CONTEXT: READ_BOUND lines, and READ_BYTE_BOUND bytes of
 * its text as the caller's Read prints it. MEASURED on the cursor seat's
 * Claude Code transcript: each read went out as a Read with only file_path,
 * so Claude Code read the whole file (a 100,627-character module twice
 * within 2 s, then 53 KB and 99 KB more); the conversation went from about
 * 90k to 155k tokens in five seconds and compacted, every three minutes.
 * That module's code runs about 55 characters a line (1,836 lines), so 400
 * lines is about 22 KB, and 24,000 bytes some 6,000 tokens by the bridge's
 * bytes-over-four estimate: room for a function and its neighbours, and a
 * dozen reads before a compaction where the whole file allowed two. Lines
 * alone do not bound a file of long lines (the seat's 99 KB WEB.md has 339),
 * and MEASURED on Claude Code 2.1.284, its Read prints a 9,014-character
 * line whole, so the byte bound holds there.
 */
const READ_BOUND = 400;
const READ_BYTE_BOUND = 24_000;

// agent.v1.ReadArgs.offset = 4 (int32) and .limit = 5 (uint32), and
// ReadSuccess.range_applied = 8 (bool), from Cursor's CLI bundles
// (2026.04.14, 2026.05.20): the vendored proto predates all three, so the
// read's range arrives as unknown fields and range_applied goes out as one.
const READ_OFFSET_FIELD = 4;
const READ_LIMIT_FIELD = 5;
const READ_RANGE_APPLIED_FIELD = 8;

/** A varint field the vendored proto cannot name, read off the message's unknown fields. */
function unknownVarint(m: ReadArgs, no: number, read: (r: BinaryReader) => number): number | undefined {
  const f = m.$unknown?.find((u) => u.no === no && u.wireType === WireType.Varint);
  return f && read(new BinaryReader(f.data));
}

/** A read's own range. */
interface ReadRange {
  /** The first line, 1-based as Cursor's CLI reads it (0 reads as 1); negative counts from the end. */
  offset?: number;
  /** The lines asked for; none reads to the end. */
  limit?: number;
}

function readRange(a: ReadArgs): ReadRange {
  const offset = unknownVarint(a, READ_OFFSET_FIELD, (r) => r.int32());
  const limit = unknownVarint(a, READ_LIMIT_FIELD, (r) => r.uint32()) || undefined;
  return { ...(offset !== undefined && { offset }), ...(limit !== undefined && { limit }) };
}

/** The lines a read goes out for when no byte bound stops it first. */
const readWant = (r: ReadRange): number => Math.min(r.limit ?? Infinity, READ_BOUND);

/**
 * THE CLIENT'S READ IS WHAT FILLS ITS CONTEXT. The caller's Read runs on the
 * client, which keeps its output in its conversation before the bridge sees
 * it, and Read takes no byte limit: the one lever on that side is how many
 * lines the Read asks for. So the bridge measures the file on its own disk
 * (the caller's disk, where the bridge runs beside the client as helm runs
 * it) and asks for no more lines than fit READ_BYTE_BOUND as Read prints
 * them. It reads the file only to count bytes; none of it goes to the client
 * or to Cursor. A read whose fit it cannot prove (a relative path, which it
 * cannot place; a file not on its disk, not a regular file, or unreadable; a
 * window past limits.readScanBytes) FAILS CLOSED: the client is never called,
 * and the read's error names the route (readPlan). The seat's reads are all
 * absolute local paths, and Claude Code's Read takes only absolute paths.
 */
const READ_SCAN_CHUNK = 1024 * 1024;

/** Bytes of one line as Claude Code's Read prints it: "N<tab>text<newline>". */
const printedBytes = (n: number, bytes: number): number => String(n).length + 2 + bytes;

/** How a read of a file fits READ_BYTE_BOUND (fitToBytes). */
type ReadFit = { lines: number } | { hugeLine: number } | { unknown: string } | null;

/**
 * How far a read of `path` from line `first` gets within READ_BYTE_BOUND:
 * { lines } when the bound stops it before `want` lines and the file's end;
 * { hugeLine } when line `first` alone passes the bound; { unknown } with
 * why, when the fit cannot be proven here; null when nothing needs cutting.
 */
function fitToBytes(path: string, first: number, want: number): ReadFit {
  if (!isAbsolute(path)) return { unknown: "it is not an absolute path on this machine" };
  let fd: number;
  try {
    if (!statSync(path).isFile()) return { unknown: "it is not a regular file" };
    fd = openSync(path, "r");
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code;
    return { unknown: code === "ENOENT" ? "no file is at that path on this machine" : `it cannot be read here (${code ?? e})` };
  }
  try {
    const chunk = Buffer.alloc(READ_SCAN_CHUNK);
    let line = 1, bytes = 0, used = 0, taken = 0;
    for (let pos = 0, n = 0; ; pos += n) {
      if (pos >= limits.readScanBytes) {
        return { unknown: `the lines it asks for lie past the first ${limits.readScanBytes} bytes the bridge measures` };
      }
      n = readSync(fd, chunk, 0, Math.min(chunk.length, limits.readScanBytes - pos), pos);
      if (n === 0) return null;   // the file ended within the bound
      const view = chunk.subarray(0, n);
      for (let i = 0; i < n;) {
        const nl = view.indexOf(10, i);
        bytes += (nl === -1 ? n : nl) - i;
        if (line >= first && used + printedBytes(line, bytes) > READ_BYTE_BOUND) {
          return taken ? { lines: taken } : { hugeLine: line };
        }
        if (nl === -1) break;
        if (line >= first) {
          used += printedBytes(line, bytes);
          if (++taken === want) return null;
        }
        line++;
        bytes = 0;
        i = nl + 1;
      }
    }
  } catch (e) {
    return { unknown: `it cannot be read here (${(e as NodeJS.ErrnoException).code ?? e})` };
  } finally {
    closeSync(fd);
  }
}

/** A read the bridge answers itself: the read's error, naming the route (BridgeAnswer). */
function readError(a: ReadArgs, error: string): BridgeAnswer {
  return { answer: [{ case: "readResult", value: create(ReadResultSchema, {
    result: { case: "error", value: create(ReadErrorSchema, { path: a.path, error }) },
  }) }] };
}

/**
 * A native read as the caller's Read, or as the bridge's own answer when
 * Read cannot do it within the bounds: a read from the end (Read counts from
 * the start, and marking the first lines as the last would be the wrong
 * slice as a success); a line that alone passes READ_BYTE_BOUND (Read cannot
 * read part of a line); a read whose fit the bridge cannot prove (it fails
 * closed). Each is that one request's error naming the route, never a
 * refusal of the tool.
 */
function readPlan(a: ReadArgs): CallerArgs | BridgeAnswer {
  const r = readRange(a);
  const file = shellWord(a.path);
  if (r.offset !== undefined && r.offset < 0) {
    const n = -r.offset;
    trace(`native readArgs asks for the last ${n} lines: answered with the tail route`);
    return readError(a, `Reading from the end of a file is not available here: the caller's Read counts lines from the start. ` +
      `With the Shell tool, read the last ${n} lines with tail -n ${n} ${file}${r.limit && r.limit < n ? ` | head -n ${r.limit}` : ""}, ` +
      `and count all of them with wc -l ${file}.`);
  }
  const first = Math.max(r.offset ?? 1, 1);
  const fit = fitToBytes(a.path, first, readWant(r));
  if (fit && "unknown" in fit) {
    trace(`native readArgs cannot be measured (${fit.unknown}): answered with the Shell route`);
    return readError(a, `The bridge cannot measure ${a.path}: ${fit.unknown}, so this read could bring more than ` +
      `${READ_BYTE_BOUND} bytes into the caller's context and was not made. ` +
      `${isAbsolute(a.path) ? "With the Shell tool, read" : "Read it by its absolute path on this machine, or with the Shell tool read"} ` +
      `the lines with sed -n '${first},${first + readWant(r) - 1}p' ${file} | cut -b 1-${READ_BYTE_BOUND} | head -c ${READ_BYTE_BOUND}.`);
  }
  if (fit && "hugeLine" in fit) {
    const n = fit.hugeLine;
    trace(`native readArgs line ${n} passes ${READ_BYTE_BOUND} bytes: answered with the Shell route`);
    return readError(a, `Line ${n} of ${a.path} is longer than the ${READ_BYTE_BOUND} bytes one read brings into the caller's context, ` +
      `so it was not read. With the Shell tool, read it in parts with sed -n '${n}p' ${file} | cut -b 1-${READ_BYTE_BOUND}, ` +
      `then cut -b ${READ_BYTE_BOUND + 1}-${2 * READ_BYTE_BOUND} and on, and the lines after it with sed -n '${n + 1},${n + READ_BOUND}p' ${file}.`);
  }
  const limit = fit ? fit.lines : readWant(r);
  trace(`native readArgs ${r.offset !== undefined || r.limit !== undefined ? "carries a range" : "carries no range"}:`,
    `Read offset=${first} limit=${limit}${fit ? ` (the lines that fit ${READ_BYTE_BOUND} bytes)` : ""}`);
  return { need: { file_path: a.path, limit, ...(first > 1 && { offset: first }) } };
}

/**
 * The client's Read output held to READ_BYTE_BOUND bytes on its way to
 * Cursor: cut after its last whole line within the bound ("lines"), or, when
 * its first line alone passes it, at the last whole character within it
 * ("line"). The measure keeps the client's read within the bound; this holds
 * Cursor's side too when the file grew between the measure and the read.
 */
function capRead(text: string): { text: string; cut: "lines" | "line" | null } {
  const buf = Buffer.from(text, "utf8");
  if (buf.length <= READ_BYTE_BOUND) return { text, cut: null };
  const nl = buf.lastIndexOf(10, READ_BYTE_BOUND);
  if (nl > 0) return { text: buf.subarray(0, nl).toString("utf8"), cut: "lines" };
  let end = READ_BYTE_BOUND;
  while (end > 0 && (buf[end]! & 0xc0) === 0x80) end--;
  return { text: buf.subarray(0, end).toString("utf8"), cut: "line" };
}

/**
 * What the bridge adds to a read a bound stopped: where it stopped, which
 * bound stopped it, and the Shell route to the rest. Claude Code's Read
 * prints "N<tab>text" per line and no total, so the note names `wc -l`
 * rather than a count. A read that got every line it asked for gets nothing.
 */
function readNote(a: ReadArgs, r: ReadRange, limit: number, text: string, cut: "lines" | "line" | null): string {
  const lines = [...text.matchAll(/^ *(\d+)\t/gm)];
  const file = shellWord(a.path);
  const find = `find lines with git grep -n PATTERN ${file}, and count them with wc -l ${file}.]`;
  const bytes = `the ${READ_BYTE_BOUND} bytes the bridge brings per read`;
  if (cut === "line" && lines[0]) {
    const n = Number(lines[0][1]);
    const held = utf8Bytes(text.slice(lines[0].index! + lines[0][0].length));
    return `\n\n[This read stops partway through line ${n}: it holds the first ${held} bytes of that line, all ${bytes} allow. ` +
      `With the Shell tool, read more of it with sed -n '${n}p' ${file} | cut -b ${held + 1}-${held + READ_BYTE_BOUND}, ` +
      `the lines after it with sed -n '${n + 1},${n + READ_BOUND}p' ${file}, ${find}`;
  }
  const last = Number(lines.at(-1)?.[1] ?? 0);
  if (cut && !last) return `\n\n[This read was cut to ${bytes}. With the Shell tool, ${find}`;
  const why = cut ? `the text was cut to ${bytes}`
    : lines.length < limit || limit >= (r.limit ?? Infinity) ? null
    : limit < readWant(r) ? `the next line would pass ${bytes}`
    : `the bridge brings at most ${READ_BOUND} lines per read`;
  if (!why) return "";
  return `\n\n[This read stops at line ${last}: ${why}, and the file may go on. With the Shell tool, ` +
    `read the next lines with sed -n '${last + 1},${last + READ_BOUND}p' ${file}, ${find}`;
}

/**
 * Each native the bridge runs as a caller tool: the tool, its arguments, and
 * how the client's output becomes the native's own result. Natives not here
 * (delete, background shells, stdin writes, and the rest) keep the refusal:
 * each is destructive or has no caller tool whose output gives its result
 * honestly. A read maps to Read only, never to Bash cat: a caller that
 * registered Bash but no Read chose that, and the refusal names Bash. A read
 * goes out with a limit (readPlan), so a Read whose schema takes none keeps
 * the refusal rather than bring a whole file into the caller's context.
 */
const NATIVE_ROUTES: { [C in NativeCase]: NativeRoute<ExecArgs<C>> } = {
  readArgs: {
    tool: "Read",
    carriesText: true,
    call: readPlan,
    answer(a, out, note, sent) {
      const o = outcome(out);
      if (!o.ok) {
        const error = create(ReadErrorSchema, { path: a.path, error: o.why + note });
        return [{ case: "readResult", value: create(ReadResultSchema, { result: { case: "error", value: error } }) }];
      }
      const r = readRange(a);
      const { text, cut } = capRead(o.text);
      const more = readNote(a, r, Number(sent.limit), text, cut);
      const success = create(ReadSuccessSchema, {
        path: a.path, totalLines: lineCount(text), truncated: more !== "", output: { case: "content", value: text + more + note },
      });
      // Cursor's CLI sets range_applied exactly when it cut the content to
      // the read's range, so Cursor does not cut those lines again. A ranged
      // Read's output is already those lines; a read with no range has none
      // to apply, and goes out without it as the CLI's does.
      if (r.offset !== undefined || r.limit !== undefined) {
        success.$unknown = [{
          no: READ_RANGE_APPLIED_FIELD, wireType: WireType.Varint, data: new BinaryWriter().bool(true).finish(),
        }];
      }
      return [{ case: "readResult", value: create(ReadResultSchema, { result: { case: "success", value: success } }) }];
    },
  },
  shellArgs: { tool: "Bash", carriesText: true, call: inDirectory, answer: shellAnswer },
  shellStreamArgs: { tool: "Bash", carriesText: true, streamed: true, call: inDirectory, answer: shellStreamAnswer },
  grepArgs: {
    tool: "Grep",
    carriesText: false,
    // Cursor's sort and sort_ascending have no Grep field and are dropped.
    call: (a) => ({
      need: { pattern: a.pattern, output_mode: grepMode(a) },
      may: {
        path: a.path, glob: a.glob, "-B": a.contextBefore, "-A": a.contextAfter, "-C": a.context,
        "-i": a.caseInsensitive, type: a.type, head_limit: a.headLimit, multiline: a.multiline,
        "-n": grepMode(a) === "content" || undefined,
      },
    }),
    answer(a, out) {
      const grep = (result: GrepResult["result"]): ExecAnswer[] => [{ case: "grepResult", value: create(GrepResultSchema, { result }) }];
      const o = outcome(out);
      if (!o.ok) return grep({ case: "error", value: create(GrepErrorSchema, { error: o.why }) });
      const found = GREP_READERS[grepMode(a)]?.(o.text, a);
      if (!found) {
        return grep({ case: "error", value: create(GrepErrorSchema, {
          error: `The caller's Grep answered in a form the bridge cannot split into ${grepMode(a)} results; its whole answer:\n${o.text}`,
        }) });
      }
      // Claude Code prints each path as the search was given it, so the
      // workspace key is empty rather than a root the paths would be joined to.
      return grep({ case: "success", value: create(GrepSuccessSchema, {
        pattern: a.pattern, path: a.path ?? "", outputMode: grepMode(a),
        workspaceResults: { "": create(GrepUnionResultSchema, { result: found }) },
      }) });
    },
  },
  writeArgs: {
    tool: "Write",
    carriesText: false,
    call: (a) => (a.fileBytes.length ? null : { need: { file_path: a.path, content: a.fileText } }),
    answer(a, out) {
      // A write's success carries no text, so it is a success only when the
      // client said so ("File created successfully at", "The file ... has
      // been updated successfully"); any other words are its error.
      const o = outcome(out);
      const why = !o.ok ? o.why : /\bsuccessfully\b/.test(o.text) ? null : o.text;
      const result = why === null
        ? { case: "success" as const, value: create(WriteSuccessSchema, {
            path: a.path, linesCreated: lineCount(a.fileText), fileSize: utf8Bytes(a.fileText),
            ...(a.returnFileContentAfterWrite && { fileContentAfterWrite: a.fileText }),
          }) }
        : { case: "error" as const, value: create(WriteErrorSchema, { path: a.path, error: why }) };
      return [{ case: "writeResult", value: create(WriteResultSchema, { result }) }];
    },
  },
  lsArgs: {
    // Bash, not Glob: Glob lists files only, and a listing's tree needs its
    // directories. `ls -1Ap` prints one name per line with "/" on each
    // directory, and LS_END proves the whole listing came back. Cursor's
    // ignore globs are not applied.
    tool: "Bash",
    carriesText: false,
    call: (a) => ({ need: { command: `ls -1Ap -- ${shellWord(a.path || ".")} && echo ${shellWord(LS_END)}` } }),
    answer(a, out) {
      const o = outcome(out);
      const names = o.ok ? o.text.split("\n").filter(Boolean) : [];
      const why = !o.ok ? o.why : names.at(-1) === LS_END ? null : o.text;
      if (why !== null) {
        return [{ case: "lsResult", value: create(LsResultSchema, {
          result: { case: "error", value: create(LsErrorSchema, { path: a.path, error: why }) },
        }) }];
      }
      const entries = names.slice(0, -1);
      const files = entries.filter((n) => !n.endsWith("/"));
      const root = create(LsDirectoryTreeNodeSchema, {
        absPath: a.path,
        childrenDirs: entries.filter((n) => n.endsWith("/"))
          .map((d) => create(LsDirectoryTreeNodeSchema, { absPath: pathJoin(a.path, d.slice(0, -1)) })),
        childrenFiles: files.map((name) => create(LsDirectoryTreeNode_FileSchema, { name })),
        childrenWereProcessed: true,
        numFiles: files.length,
      });
      return [{ case: "lsResult", value: create(LsResultSchema, {
        result: { case: "success", value: create(LsSuccessSchema, { directoryTreeRoot: root }) },
      }) }];
    },
  },
  fetchArgs: {
    tool: "WebFetch",
    carriesText: true,
    call: (a) => ({ need: { url: a.url }, may: { prompt: "Return the page content." } }),
    // WebFetch's answer carries no HTTP status, so none is reported.
    answer(a, out, note) {
      const o = outcome(out);
      const result = o.ok
        ? { case: "success" as const, value: create(FetchSuccessSchema, { url: a.url, content: o.text + note }) }
        : { case: "error" as const, value: create(FetchErrorSchema, { url: a.url, error: o.why + note }) };
      return [{ case: "fetchResult", value: create(FetchResultSchema, { result }) }];
    },
  },
};

function routeOf(c: NativeCase): NativeRoute<NativeArgs> {
  return NATIVE_ROUTES[c] as NativeRoute<NativeArgs>;
}

/**
 * The arguments that go out, fitted to the schema the caller registered for
 * its tool: a `may` field it does not declare is dropped, and a `need` field
 * it does not declare, or a field it requires that this call cannot fill,
 * means the call is not made (the refusal names the tool instead). A schema
 * with no properties object is taken as it stands.
 */
function fitSchema(tool: McpToolDefinition, args: CallerArgs): Record<string, unknown> | null {
  const schema = decodeMcpArgValue(tool.inputSchema);
  const { properties, required } = (schema && typeof schema === "object" ? schema : {}) as { properties?: unknown; required?: unknown };
  const declared = properties && typeof properties === "object" ? new Set(Object.keys(properties)) : null;
  if (declared && Object.keys(args.need).some((k) => !declared.has(k))) return null;
  const may = Object.entries(args.may ?? {}).filter(([k, v]) => v !== undefined && (!declared || declared.has(k)));
  const out = { ...args.need, ...Object.fromEntries(may) };
  return (Array.isArray(required) ? required : []).every((k) => k in out) ? out : null;
}

/**
 * A native exec as the caller tool call that does its job; as the native's
 * own answer frames when the bridge answers it itself (BridgeAnswer); or
 * null when it stays refused.
 */
function translateNative(execMsg: ExecServerMessage, mcpTools: McpToolDefinition[]): PendingExec | Buffer[] | null {
  const { case: c, value } = execMsg.message;
  if (!c || !Object.hasOwn(NATIVE_ROUTES, c)) return null;
  const route = routeOf(c as NativeCase);
  const tool = mcpTools.find((t) => (t.toolName || t.name) === route.tool);
  if (!tool) return null;
  const call = route.call(value as NativeArgs);
  if (call && "answer" in call) {
    trace(`native ${c} is answered by the bridge, with no call to the caller's ${route.tool}`);
    return answerFrames(route, { id: execMsg.id, execId: execMsg.execId }, call.answer);
  }
  const args = call && fitSchema(tool, call);
  if (!args) return null;
  const cursorToolCallId = (value as NativeArgs).toolCallId || crypto.randomUUID();
  trace(`native ${c} runs as the caller's ${route.tool}`);
  return {
    execId: execMsg.execId,
    execMsgId: execMsg.id,
    toolCallId: clientToolCallId(cursorToolCallId),
    cursorToolCallId,
    toolName: route.tool,
    decodedArgs: JSON.stringify(args),
    native: { case: c as NativeCase, args: value as NativeArgs },
  };
}

/** Cursor's answer to a translated native: its own result for the client's output. */
function nativeFrames(exec: PendingExec, out: string | undefined, note: string): Buffer[] {
  const route = routeOf(exec.native!.case);
  const sent = JSON.parse(exec.decodedArgs) as Record<string, unknown>;
  return answerFrames(route, { id: exec.execMsgId, execId: exec.execId }, route.answer(exec.native!.args, out, note, sent));
}

/** A native's answers as frames to Cursor, then streamClose when it streams. */
function answerFrames(route: NativeRoute<NativeArgs>, target: { id: number; execId: string }, answers: ExecAnswer[]): Buffer[] {
  const frames = answers.map((a) => execClientFrame(target, a.case, a.value));
  if (!route.streamed) return frames;
  const control = create(ExecClientControlMessageSchema, {
    message: { case: "streamClose", value: create(ExecClientStreamCloseSchema, { id: target.id }) },
  });
  const clientMessage = create(AgentClientMessageSchema, {
    message: { case: "execClientControlMessage", value: control },
  });
  return [...frames, frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage))];
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
  const bridge = seams.spawn(accessToken);
  bridge.write(frameConnectMessage(payload.requestBytes));
  const heartbeatTimer = setInterval(() => {
    bridge.write(makeHeartbeatBytes());
  }, 5_000);
  const peeked = await peekFirstFrame(bridge, limits.firstFrameMs);
  const continued = payload.continued;
  if (peeked.refusal) {
    clearInterval(heartbeatTimer);
    bridge.end();
    console.error(`[bridge] upstream refusal on first frame: ${peeked.refusal.code} -> 429`);
    // The client retries a 429; the conversation it would have continued waits for it.
    if (continued) keptConversations.set(continued.kept.chain.at(-1)!, continued.kept);
    return refusalResponse(peeked.refusal);
  }
  if (continued && peeked.error) {
    // Cursor would not continue the kept conversation: answer this request
    // from a replay of the whole history instead, and say so.
    clearInterval(heartbeatTimer);
    bridge.end();
    const payload = continued.replay();
    console.error(`[bridge] Cursor would not continue conversation ${short(continued.kept.id)} (${peeked.error.message}); new Cursor conversation ${short(payload.conversationId)}: replaying the history (${kb(payload.blobStore)})`);
    return startRun(payload, accessToken, modelId, stream, promptTokens);
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
    conversationId: payload.conversationId,
    conversationRuns: continued ? continued.kept.runs + 1 : 1,
    checkpoint: continued?.kept.state ?? null,
    chain: payload.chain,
    ended: null,
    sent: { request: payload.requestBytes.length, answers: 0, blobs: 0, blobBytes: 0 },
  };
  trace(`run ${run.id} started (${modelId})`);
  if (continued) {
    console.error(`[bridge] run ${run.id} continues Cursor conversation ${short(payload.conversationId)} (run ${run.conversationRuns} of it): sent ${payload.requestBytes.length} bytes, its checkpoint and the new turn; no history replayed`);
  }
  return respond(run, stream, () => listen(run, peeked));
}

/** Write a frame to Cursor on this run, counting it as an answer (Run.sent). */
function sendOnRun(run: Run, data: Uint8Array): void {
  if (run.sent) run.sent.answers += data.length;
  run.bridge.write(data);
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
        (data) => sendOnRun(run, data),
        (text, isThinking) => emitText(run, text, Boolean(isThinking)),
        (exec) => onExec(run, exec),
        (counts) => countTokens(run, counts),
        (state) => { run.checkpoint = state; },
        (bytes) => {
          if (!run.sent) return;
          run.sent.blobs++;
          run.sent.blobBytes += bytes;
        },
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
  if (run.sent) {
    const { request, answers, blobs, blobBytes } = run.sent;
    console.error(`[bridge] run ${run.id} closed: sent Cursor ${request + answers} bytes (the run request ${request}, then ${answers} in answers, ${blobs} blob fetches of ${blobBytes})`);
  }
  keepConversation(run);
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
  run.ended ??= why;
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
  // on a paused run: it rides the last result instead of vanishing. A
  // translated native's result carries it only where the native's answer has
  // text (NativeRoute.carriesText): a listing, a grep or a write has none.
  const carries = (i: number): boolean => answered[i] !== undefined
    && (!run.awaiting[i]!.native || routeOf(run.awaiting[i]!.native!.case).carriesText);
  let last = answered.length - 1;
  while (last >= 0 && !carries(last)) last--;
  if (userText && last < 0) {
    console.error(`[bridge] run ${run.id}: user text with no matching tool result was dropped`);
  }
  const note = userText ? `\n\n<user_message>\n${userText}\n</user_message>` : "";
  for (const [i, exec] of run.awaiting.entries()) {
    const result = answered[i];
    const ride = i === last ? note : "";
    if (!result) console.error(`[bridge] run ${run.id}: no result for tool call ${exec.toolCallId}`);
    if (exec.native) {
      for (const frame of nativeFrames(exec, result?.content, ride)) sendOnRun(run, frame);
      continue;
    }
    const mcpResult = result
      ? create(McpResultSchema, {
          result: {
            case: "success",
            value: create(McpSuccessSchema, {
              content: [
                create(McpToolResultContentItemSchema, {
                  content: {
                    case: "text",
                    value: create(McpTextContentSchema, { text: result.content + ride }),
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
            value: create(McpErrorSchema, { error: NOT_PROVIDED }),
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
    sendOnRun(run, frameConnectMessage(toBinary(AgentClientMessageSchema, clientMessage)));
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
 * call/result id, call name and arguments, and every tool's name, description
 * and forwarded parameter schema, over BYTES_PER_TOKEN. The default system
 * and continuation prompts count when the bridge synthesizes them. It counts
 * payload fields, never the JSON the bridge wraps them in for Cursor.
 */
function requestTokens(
  messages: OpenAIMessage[],
  tools: OpenAIToolDef[],
  stateless = true,
): number {
  let bytes = 0;
  for (const m of messages) {
    bytes += utf8Bytes(contentToText(m.content));
    if (m.role === "tool") bytes += utf8Bytes(m.tool_call_id ?? "");
    for (const call of m.tool_calls ?? []) {
      bytes += utf8Bytes(call.id) + utf8Bytes(call.function.name)
        + utf8Bytes(call.function.arguments ?? "");
    }
  }
  for (const { function: fn } of tools) {
    const schema = fn.parameters && typeof fn.parameters === "object"
      ? fn.parameters
      : EMPTY_TOOL_SCHEMA;
    bytes += utf8Bytes(fn.name) + utf8Bytes(fn.description ?? "")
      + utf8Bytes(JSON.stringify(schema));
  }
  // The stateless path adds these prompts. A paused-run resume sends tool
  // results directly, so neither exists on that path.
  if (stateless) {
    if (!messages.some((m) => m.role === "system")) {
      bytes += utf8Bytes(DEFAULT_SYSTEM_PROMPT);
    }
    const last = messages.filter((m) => m.role !== "system").at(-1);
    if (!last || last.role !== "user" || !contentToText(last.content).trim()) {
      bytes += utf8Bytes(CONTINUE_PROMPT);
    }
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
          sink.outBytes += utf8Bytes(exec.toolCallId) + utf8Bytes(exec.toolName)
            + utf8Bytes(exec.decodedArgs);
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
      sink.outBytes += utf8Bytes(exec.toolCallId) + utf8Bytes(exec.toolName)
        + utf8Bytes(exec.decodedArgs);
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
export const __test = { IDLE_TIMEOUT_S, IDLE_MARGIN_MS, stallWindow, requestTokens, handleChatCompletion, peekFirstFrame, connectRefusal, handleExecMessage, clientToolCallId, respond, listen, findPausedRun, takePausedRun, stampConversation, resumeRun, pausedRuns, limits, toHistory, buildCursorRequest, parseMessages, seams, keptConversations, messageChain, continuation, buildContinuation, mcpInstructionsFor };
