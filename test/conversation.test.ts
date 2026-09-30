// helm arm (2026-09-30, task/3817): CURSOR KEEPS ITS OWN CONVERSATION ACROSS
// THE CLIENT'S TURNS. Cursor's own CLI stores each checkpoint Cursor sends
// (conversationCheckpointUpdate) with the blobs Cursor writes, and sends that
// checkpoint back under the same conversation id with only the new user
// message. The bridge threw the checkpoint away and replayed the whole
// history into a new conversation (random id, every message a root-prompt
// blob) at the start of every run. MEASURED on the cursor seat's bridge.log:
// on that cold replay Cursor counted the same ~140k history 107k, 214k, 426k
// or 905k, over its 256k window in 16 of 24 run starts, and 13 of 15 turns
// over 256k made no tool call. A request that extends a kept conversation by
// its reply and new user text now sends only that text; anything else (a
// compaction, a /clear, an edited history, a run that did not close cleanly)
// replays, and the bridge log says why.
//
// And the caller's tools are named in Cursor's MCP instructions on every run:
// after a compaction the cursor seat decided "the Monitor tool is
// unavailable" (it never called GetDynamicTools) and its beacon stayed dead.
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { create, fromBinary } from "@bufbuild/protobuf";
import { BinaryWriter, WireType } from "@bufbuild/protobuf/wire";
import {
  AgentClientMessageSchema,
  AgentServerMessageSchema,
  ConversationStateStructureSchema,
  ConversationTokenDetailsSchema,
  ExecServerMessageSchema,
  KvServerMessageSchema,
  GetBlobArgsSchema,
  SetBlobArgsSchema,
  RequestContextArgsSchema,
  McpToolDefinitionSchema,
  type AgentRunRequest,
  type RequestContextResult,
} from "../src/proto/agent_pb";
import { __test } from "../src/proxy";
import { fakeBridge, frame, mcpCall, readSse, textDelta } from "./helpers";

type Msg = { role: "system" | "user" | "assistant" | "tool"; content: unknown; tool_call_id?: string; tool_calls?: unknown[] };
const TOOLS = [
  { type: "function", function: { name: "Bash", description: "Executes a given bash command. It runs in a shell.", parameters: { type: "object", properties: { command: { type: "string" } } } } },
  { type: "function", function: { name: "Monitor", description: "Start a background monitor that streams events from a long-running script. Each stdout line is a notification.", parameters: { type: "object", properties: { command: { type: "string" } } } } },
];

const saved = { ...__test.limits };
const realSpawn = __test.seams.spawn;
let bridges: ReturnType<typeof fakeBridge>[] = [];
let logged: string[] = [];
const realError = console.error;

beforeEach(() => {
  bridges = [];
  logged = [];
  __test.seams.spawn = () => {
    const fb = fakeBridge();
    bridges.push(fb);
    return fb.bridge as never;
  };
  console.error = (...parts: unknown[]) => { logged.push(parts.map(String).join(" ")); };
});

afterEach(() => {
  console.error = realError;
  __test.seams.spawn = realSpawn;
  Object.assign(__test.limits, saved);
  __test.keptConversations.clear();
  __test.pausedRuns.clear();
});

const blobId = (s: string) => new Uint8Array(Buffer.from(s.padEnd(32, "x")));

/** Cursor's checkpoint after a step: its own turns, and a field this proto cannot read (22 = "cli"). */
function checkpointFrame(turn: string, used = 20_000): Buffer {
  const state = create(ConversationStateStructureSchema, {
    turns: [blobId(turn)],
    tokenDetails: create(ConversationTokenDetailsSchema, { usedTokens: used, maxTokens: 256_000 }),
  });
  state.$unknown = [{ no: 22, wireType: WireType.LengthDelimited, data: new BinaryWriter().string("cli").finish() }];
  return frame(create(AgentServerMessageSchema, { message: { case: "conversationCheckpointUpdate", value: state } }));
}

/** Cursor storing a turn blob (KV setBlobArgs). */
function setBlobFrame(id: string, data: string): Buffer {
  return frame(create(AgentServerMessageSchema, {
    message: {
      case: "kvServerMessage",
      value: create(KvServerMessageSchema, {
        id: 1,
        message: { case: "setBlobArgs", value: create(SetBlobArgsSchema, { blobId: blobId(id), blobData: new TextEncoder().encode(data) }) },
      }),
    },
  }));
}

/** Cursor asking for a blob back (KV getBlobArgs). */
function getBlobFrame(id: string): Buffer {
  return frame(create(AgentServerMessageSchema, {
    message: {
      case: "kvServerMessage",
      value: create(KvServerMessageSchema, { id: 2, message: { case: "getBlobArgs", value: create(GetBlobArgsSchema, { blobId: blobId(id) }) } }),
    },
  }));
}

function endStream(code: string): Buffer {
  const body = new TextEncoder().encode(JSON.stringify({ error: { code, message: "x" } }));
  const head = Buffer.alloc(5);
  head[0] = 0b10;
  head.writeUInt32BE(body.length, 1);
  return Buffer.concat([head, Buffer.from(body)]);
}

/** The run request a fake bridge was sent first. */
function runRequest(fb: ReturnType<typeof fakeBridge>): { req: AgentRunRequest; bytes: number } {
  const w = fb.writes[0]!;
  const msg = fromBinary(AgentClientMessageSchema, w.subarray(5));
  return { req: msg.message.value as AgentRunRequest, bytes: w.length };
}

const text = (req: AgentRunRequest) => (req.action!.action.value as { userMessage: { text: string } }).userMessage.text;

function ask(messages: Msg[]) {
  return __test.handleChatCompletion({ model: "grok-4.7-high", stream: true, tools: TOOLS, messages } as never, "tok");
}

/** One whole turn: the request, Cursor's frames, a clean close; the client's reading of it. */
async function turn(messages: Msg[], ...frames: Buffer[]) {
  const pending = ask(messages);
  const fb = bridges.at(-1)!;
  fb.feed(...frames);
  fb.close(0);
  return { fb, sse: await readSse(await pending) };
}

const SYSTEM: Msg = { role: "system", content: "You are a coding agent." };
const FIRST: Msg[] = [SYSTEM, { role: "user", content: "The codeword is TEAL-4417. Reply only: noted." }];

describe("a kept Cursor conversation", () => {
  test("the next turn sends Cursor only the new user text, with the last checkpoint and the same conversation id", async () => {
    const one = await turn(FIRST, setBlobFrame("t1", "turn one"), checkpointFrame("t1"), frame(textDelta("noted")));
    expect(one.sse.content).toBe("noted");
    const first = runRequest(one.fb).req;
    // the system prompt; the user message is the run's action
    expect(first.conversationState!.rootPromptMessagesJson.length).toBe(1);

    const second = [...FIRST, { role: "assistant" as const, content: "noted" }, { role: "user" as const, content: "What is the codeword?" }];
    const two = await turn(second, frame(textDelta("TEAL-4417")));
    const { req } = runRequest(two.fb);
    expect(req.conversationId).toBe(first.conversationId);
    expect(text(req)).toBe("What is the codeword?");
    // Cursor's own checkpoint went back as it came: its turns and the field
    // this proto cannot read
    expect(req.conversationState!.turns.map((t) => Buffer.from(t).toString())).toEqual([Buffer.from(blobId("t1")).toString()]);
    expect(req.conversationState!.$unknown?.map((u) => u.no)).toEqual([22]);
    expect(two.sse.content).toBe("TEAL-4417");
    expect(logged.some((l) => l.includes(`continues Cursor conversation ${first.conversationId.slice(0, 8)}`))).toBe(true);
  });

  test("Cursor gets back every blob it stored, and the replayed root prompt, from the kept store", async () => {
    const one = await turn(FIRST, setBlobFrame("t1", "turn one"), checkpointFrame("t1"), frame(textDelta("noted")));
    const root = runRequest(one.fb).req.conversationState!.rootPromptMessagesJson[0]!;
    const second = [...FIRST, { role: "assistant" as const, content: "noted" }, { role: "user" as const, content: "next" }];
    const pending = ask(second);
    const fb = bridges.at(-1)!;
    fb.feed(getBlobFrame("t1"));
    const rootAsk = frame(create(AgentServerMessageSchema, {
      message: { case: "kvServerMessage", value: create(KvServerMessageSchema, { id: 3, message: { case: "getBlobArgs", value: create(GetBlobArgsSchema, { blobId: root }) } }) },
    }));
    fb.feed(rootAsk, frame(textDelta("ok")));
    fb.close(0);
    await readSse(await pending);
    const served = fb.writes.slice(1).map((w) => fromBinary(AgentClientMessageSchema, w.subarray(5)))
      .filter((m) => m.message.case === "kvClientMessage")
      .map((m) => (m.message.value as { message: { value: { blobData?: Uint8Array } } }).message.value.blobData);
    expect(new TextDecoder().decode(served[0]!)).toBe("turn one");
    expect(JSON.parse(new TextDecoder().decode(served[1]!))).toEqual({ role: "system", content: "You are a coding agent." });
    expect(logged.some((l) => l.includes("does not hold it"))).toBe(false);
  });

  test("a continued conversation carries through a tool round trip and is kept again for the turn after", async () => {
    const one = await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const id = runRequest(one.fb).req.conversationId;
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "run echo hi" }];
    const pending = ask(second);
    const fb = bridges.at(-1)!;
    fb.feed(checkpointFrame("t2"), frame(mcpCall(1, "call-1", "echo hi")));
    const paused = await readSse(await pending);
    expect(paused.finish).toBe("tool_calls");
    const third: Msg[] = [...second,
      { role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function", function: { name: "Bash", arguments: "{\"command\":\"echo hi\"}" } }] },
      { role: "tool", tool_call_id: "call-1", content: "hi\n" }];
    const resumed = ask(third);
    fb.feed(checkpointFrame("t3"), frame(textDelta("it printed hi")));
    fb.close(0);
    expect((await readSse(await resumed)).content).toBe("it printed hi");
    expect(bridges.length).toBe(2);   // the tool round trip resumed the same run

    const fourth: Msg[] = [...third, { role: "assistant", content: "it printed hi" }, { role: "user", content: "thanks" }];
    const four = await turn(fourth, frame(textDelta("welcome")));
    const { req } = runRequest(four.fb);
    expect(req.conversationId).toBe(id);
    expect(text(req)).toBe("thanks");
    expect(req.conversationState!.turns.map((t) => Buffer.from(t).toString())).toEqual([Buffer.from(blobId("t3")).toString()]);
  });

  test("a continued run that sends no checkpoint of its own is not kept, and the next turn replays", async () => {
    // Kept under the new history with the previous turn's checkpoint, the
    // conversation would lose this turn's user text and reply on Cursor's
    // side while the client's history carries them.
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }];
    const two = await turn(second, frame(textDelta("ok")));
    const id = runRequest(two.fb).req.conversationId;
    expect(text(runRequest(two.fb).req)).toBe("next");
    expect(__test.keptConversations.size).toBe(0);
    expect(logged.some((l) => l.includes(`Cursor conversation ${id.slice(0, 8)} is not kept, since Cursor sent no checkpoint`))).toBe(true);
    const three = await turn([...second, { role: "assistant", content: "ok" }, { role: "user", content: "and then" }], frame(textDelta("done")));
    const { req } = runRequest(three.fb);
    expect(req.conversationId).not.toBe(id);
    expect(req.conversationState!.turns.length).toBe(0);
    expect(req.conversationState!.rootPromptMessagesJson.length).toBe(5);
  });

  test("the reminders Claude Code sends with a new prompt ride the new turn's text", async () => {
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" },
      { role: "user", content: [{ type: "text", text: "<system-reminder>r</system-reminder>" }, { type: "text", text: "go" }] },
      { role: "user", content: "and more" }];
    const two = await turn(second, frame(textDelta("ok")));
    expect(text(runRequest(two.fb).req)).toBe("<system-reminder>r</system-reminder>\ngo\n\nand more");
  });

  test("cache markers and string-or-block content do not break the match", () => {
    const a = __test.messageChain([SYSTEM, { role: "user", content: "hello" }] as never);
    const b = __test.messageChain([SYSTEM, { role: "user", content: [{ type: "text", text: "hello", cache_control: { type: "ephemeral" } }] }] as never);
    expect(b).toEqual(a);
    const c = __test.messageChain([SYSTEM, { role: "user", content: "hellO" }] as never);
    expect(c[1]).toBe(a[1]);
    expect(c[2]).not.toBe(a[2]);
  });
});

describe("a replay, and the log says why", () => {
  test("a compaction's rewritten history replays into a new conversation", async () => {
    const one = await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const id = runRequest(one.fb).req.conversationId;
    const compacted: Msg[] = [SYSTEM, { role: "user", content: "This session is being continued from a previous conversation. Summary: ..." }, { role: "assistant", content: "ok" }, { role: "user", content: "go on" }];
    const two = await turn(compacted, frame(textDelta("going")));
    const { req } = runRequest(two.fb);
    expect(req.conversationId).not.toBe(id);
    expect(req.conversationState!.turns.length).toBe(0);
    expect(req.conversationState!.rootPromptMessagesJson.length).toBe(3);
    const line = logged.find((l) => l.includes(`new Cursor conversation ${req.conversationId.slice(0, 8)}`))!;
    expect(line).toContain("replaying 4 messages");
    expect(line).toContain(`no longer extends Cursor conversation ${id.slice(0, 8)}`);
    expect(line).toContain("keeps 1 of that conversation's 2 messages");
  });

  test("an edited history replays and names where it left the kept conversation", async () => {
    const long: Msg[] = [SYSTEM, { role: "user", content: "a" }, { role: "assistant", content: "b" }, { role: "user", content: "c" }];
    await turn(long, checkpointFrame("t1"), frame(textDelta("d")));
    const edited: Msg[] = [SYSTEM, { role: "user", content: "a" }, { role: "assistant", content: "b" }, { role: "user", content: "C!" }, { role: "assistant", content: "d" }, { role: "user", content: "e" }];
    await turn(edited, frame(textDelta("f")));
    expect(logged.some((l) => l.includes("keeps 3 of that conversation's 4 messages"))).toBe(true);
  });

  test("a retry of an earlier point replays and says it is a rewind", async () => {
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }];
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    await turn(second, checkpointFrame("t2"), frame(textDelta("ok")));
    await turn(FIRST, frame(textDelta("noted again")));
    expect(logged.some((l) => l.includes("all 2 of its messages are that conversation's first (a rewind or a retry)"))).toBe(true);
  });

  test("a run the client hung up on is not kept, and the next turn replays", async () => {
    const pending = ask(FIRST);
    const fb = bridges.at(-1)!;
    fb.feed(checkpointFrame("t1"), frame(textDelta("partial")));
    const resp = await pending;
    await resp.body!.cancel();
    expect(fb.state.killed).toBe(true);
    expect(__test.keptConversations.size).toBe(0);
    expect(logged.some((l) => /is not kept, since the bridge ended it \(client disconnected\)/.test(l))).toBe(true);
    const two = await turn([...FIRST, { role: "assistant", content: "partial" }, { role: "user", content: "again" }], frame(textDelta("ok")));
    expect(runRequest(two.fb).req.conversationState!.rootPromptMessagesJson.length).toBe(3);
    expect(logged.some((l) => l.includes("no Cursor conversation is kept"))).toBe(true);
  });

  test("a run Cursor ended with an error is not kept", async () => {
    const pending = ask(FIRST);
    const fb = bridges.at(-1)!;
    fb.feed(checkpointFrame("t1"), frame(textDelta("x")), endStream("internal"));
    fb.close(0);
    await (await pending).text();
    expect(__test.keptConversations.size).toBe(0);
  });

  test("a tool result after the reply is a turn the conversation never saw, so it replays", async () => {
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const odd: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "tool", tool_call_id: "x", content: "y" }, { role: "user", content: "z" }];
    const two = await turn(odd, frame(textDelta("ok")));
    expect(runRequest(two.fb).req.conversationState!.turns.length).toBe(0);
    expect(logged.some((l) => l.includes("a tool message follows its reply"))).toBe(true);
  });

  test("Cursor refusing to continue answers the same request from a replay", async () => {
    const one = await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const id = runRequest(one.fb).req.conversationId;
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }];
    const pending = ask(second);
    bridges[1]!.feed(endStream("failed_precondition"));
    await Bun.sleep(0);
    await Bun.sleep(0);
    const replay = bridges[2]!;
    replay.feed(frame(textDelta("answered")));
    replay.close(0);
    expect((await readSse(await pending)).content).toBe("answered");
    const { req } = runRequest(replay);
    expect(req.conversationId).not.toBe(id);
    expect(req.conversationState!.rootPromptMessagesJson.length).toBe(3);
    expect(logged.some((l) => l.includes(`Cursor would not continue conversation ${id.slice(0, 8)}`))).toBe(true);
  });

  test("a quota refusal on a continuation keeps the conversation for the client's retry", async () => {
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }];
    const pending = ask(second);
    bridges[1]!.feed(endStream("resource_exhausted"));
    expect((await pending).status).toBe(429);
    expect(__test.keptConversations.size).toBe(1);
    const retry = await turn(second, frame(textDelta("ok")));
    expect(text(runRequest(retry.fb).req)).toBe("next");
  });

  test("kept conversations past the count limit go oldest first", async () => {
    __test.limits.keptMax = 1;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const other: Msg[] = [SYSTEM, { role: "user", content: "a subagent's task" }];
    await turn(other, checkpointFrame("t9"), frame(textDelta("done")));
    expect(__test.keptConversations.size).toBe(1);
    const two = await turn([...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }], frame(textDelta("ok")));
    expect(runRequest(two.fb).req.conversationState!.turns.length).toBe(0);
  });
});

describe("the caller's tools, named in Cursor's MCP instructions", () => {
  test("every run's request context names each registered tool and the exact call, Monitor included", () => {
    const tools = TOOLS.map((t) => create(McpToolDefinitionSchema, { name: t.function.name, toolName: t.function.name, providerIdentifier: "opencode", description: t.function.description }));
    const exec = create(ExecServerMessageSchema, { id: 4, execId: "e4", message: { case: "requestContextArgs", value: create(RequestContextArgsSchema, {}) } });
    const sent: Uint8Array[] = [];
    __test.handleExecMessage(exec, tools, (f) => sent.push(f), () => { throw new Error("not a call"); });
    const client = fromBinary(AgentClientMessageSchema, sent[0]!.subarray(5));
    const result = (client.message.value as { message: { value: RequestContextResult } }).message.value;
    const ctx = (result.result.value as { requestContext: { mcpInstructions: Array<{ serverName: string; instructions: string }> } }).requestContext;
    expect(ctx.mcpInstructions.map((m) => m.serverName)).toEqual(["opencode"]);
    const words = ctx.mcpInstructions[0]!.instructions;
    expect(words).toContain("- Bash: Executes a given bash command.");
    expect(words).toContain("- Monitor: Start a background monitor that streams events from a long-running script.");
    expect(words).toContain(`CallDynamicTool with namespace "opencode" and toolName "Monitor"`);
    expect(words).toContain("whether or not you have called GetDynamicTools");
  });

  test("no tools, no instructions", () => {
    expect(__test.mcpInstructionsFor([])).toEqual([]);
  });
});
