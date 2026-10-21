// Regression test (2026-09-30): CURSOR KEEPS ITS OWN CONVERSATION ACROSS
// THE CLIENT'S TURNS. Cursor's own CLI stores each checkpoint Cursor sends
// (conversationCheckpointUpdate) with the blobs Cursor writes, and sends that
// checkpoint back under the same conversation id with only the new user
// message. The bridge threw the checkpoint away and replayed the whole
// history into a new conversation (random id, every message a root-prompt
// blob) at the start of every run. MEASURED on a live session's bridge log:
// on that cold replay Cursor counted the same ~140k history 107k, 214k, 426k
// or 905k, over its 256k window in 16 of 24 run starts, and 13 of 15 turns
// over 256k made no tool call. A request that extends a kept conversation by
// its reply and new user text now sends only that text; anything else (a
// compaction, a /clear, an edited history, a run that did not close cleanly)
// replays, and the bridge log says why.
//
// And the caller's tools are named in Cursor's MCP instructions on every run:
// after a compaction a live session decided "the Monitor tool is
// unavailable" (it never called GetDynamicTools) and stopped using it.
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
  InteractionUpdateSchema,
  ToolCallCompletedUpdateSchema,
  ToolCallSchema,
  ShellToolCallSchema,
  ShellResultSchema,
  ShellRejectedSchema,
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
  __test.compactAsks.clear();
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

  test("a continued run whose reply says its tools are not available is not kept, and the next turn replays", async () => {
    // MEASURED 2026-10-21 on a live session: after an upstream blip a kept
    // conversation answered "the shell returned Tool not available" with no
    // tool call reaching the bridge; kept, every later turn stayed toolless.
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "read the inbox" }];
    const said = "The shell returned Tool not available three times.";
    const two = await turn(second, checkpointFrame("t2"), frame(textDelta("The shell returned Tool not ava")), frame(textDelta("ilable three times.")));
    expect(two.sse.content).toBe(said);
    const id = runRequest(two.fb).req.conversationId;
    expect(text(runRequest(two.fb).req)).toBe("read the inbox");
    expect(__test.keptConversations.size).toBe(0);
    expect(logged.some((l) => l.includes(`Cursor conversation ${id.slice(0, 8)} is not kept, since Cursor reported tool unavailable (its reply says "tool not available"`))).toBe(true);
    const three = await turn([...second, { role: "assistant", content: said }, { role: "user", content: "try again" }], frame(textDelta("ok")));
    const { req } = runRequest(three.fb);
    expect(req.conversationId).not.toBe(id);
    expect(req.conversationState!.turns.length).toBe(0);
    expect(req.conversationState!.rootPromptMessagesJson.length).toBe(5);
  });

  test("a run whose own tool call Cursor completed as not available is not kept", async () => {
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const refused = create(ShellRejectedSchema, { command: "ls", reason: "Tool not available" });
    const completed = frame(create(AgentServerMessageSchema, {
      message: {
        case: "interactionUpdate",
        value: create(InteractionUpdateSchema, {
          message: {
            case: "toolCallCompleted",
            value: create(ToolCallCompletedUpdateSchema, {
              callId: "c1",
              toolCall: create(ToolCallSchema, {
                tool: { case: "shellToolCall", value: create(ShellToolCallSchema, { result: create(ShellResultSchema, { result: { case: "rejected", value: refused } }) }) },
              }),
            }),
          },
        }),
      },
    }));
    const two = await turn([...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "run ls" }], checkpointFrame("t2"), completed, frame(textDelta("I could not run it.")));
    const id = runRequest(two.fb).req.conversationId;
    expect(__test.keptConversations.size).toBe(0);
    expect(logged.some((l) => l.includes(`Cursor conversation ${id.slice(0, 8)} is not kept, since Cursor reported tool unavailable (its shellToolCall came back rejected "tool not available"`))).toBe(true);
  });

  test("a continued run with a checkpoint is kept, and so is one that quotes the phrase while its tool calls reach the client", async () => {
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    const second: Msg[] = [...FIRST, { role: "assistant", content: "noted" }, { role: "user", content: "next" }];
    const two = await turn(second, checkpointFrame("t2"), frame(textDelta("ok")));
    const id = runRequest(two.fb).req.conversationId;
    expect(__test.keptConversations.size).toBe(1);

    const third: Msg[] = [...second, { role: "assistant", content: "ok" }, { role: "user", content: "grep the log for it" }];
    const pending = ask(third);
    const fb = bridges.at(-1)!;
    expect(runRequest(fb).req.conversationId).toBe(id);
    fb.feed(checkpointFrame("t3"), frame(mcpCall(1, "call-1", "rg -n 'Tool not available' bridge.log")));
    expect((await readSse(await pending)).finish).toBe("tool_calls");
    const fourth: Msg[] = [...third,
      { role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function", function: { name: "Bash", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "call-1", content: "Tool not available" }];
    const resumed = ask(fourth);
    fb.feed(checkpointFrame("t4"), frame(textDelta("The log says Tool not available once.")));
    fb.close(0);
    await readSse(await resumed);
    expect(__test.keptConversations.size).toBe(1);
    expect(logged.some((l) => l.includes("tool unavailable"))).toBe(false);

    const five = await turn([...fourth, { role: "assistant", content: "The log says Tool not available once." }, { role: "user", content: "thanks" }], frame(textDelta("welcome")));
    const { req } = runRequest(five.fb);
    expect(req.conversationId).toBe(id);
    expect(text(req)).toBe("thanks");
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

// Regression test (2026-10-21): A TOOLLESS FRESH REPLAY ASKS THE CLIENT
// TO COMPACT. MEASURED on a live session: fresh replays of 376-380 messages
// (about 154,000 estimated tokens) came back toolless until a hand /compact.
// The next request on that history is answered 400 "prompt is too long" once,
// the error Claude Code answers by compacting and resending.
describe("a toolless fresh replay asks the client to compact", () => {
  const TOOLLESS = "Tool not available";
  const after = (said: string, next = "go on"): Msg[] => [...FIRST, { role: "assistant", content: said }, { role: "user", content: next }];

  test("the next request on that history is answered 400 prompt is too long, once, without reaching Cursor", async () => {
    __test.limits.compactTokens = 1;
    const one = await turn(FIRST, checkpointFrame("t1"), frame(textDelta(`The shell returned ${TOOLLESS}.`)));
    const id = runRequest(one.fb).req.conversationId;
    expect(logged.some((l) => l.includes(`Cursor conversation ${id.slice(0, 8)} is not kept, since Cursor reported tool unavailable (its reply says "tool not available", and no tool call of this run reached the bridge) on a fresh replay of 2 messages (`))).toBe(true);
    expect(logged.some((l) => l.includes("at or over the 1-token compaction floor; the next request on this history is answered 400"))).toBe(true);
    expect(__test.compactAsks.size).toBe(1);

    const spawned = bridges.length;
    const resp = await ask(after(`The shell returned ${TOOLLESS}.`));
    expect(resp.status).toBe(400);
    const body = await resp.json() as { error: { message: string; type: string; code: string } };
    expect(body.error.type).toBe("invalid_request_error");
    expect(body.error.code).toBe("context_length_exceeded");
    // the shape Claude Code parses: prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)
    const m = body.error.message.match(/prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)/i)!;
    expect(Number(m[1])).toBeGreaterThan(Number(m[2]));
    expect(Number(m[2])).toBe(1);
    expect(bridges.length).toBe(spawned);
    expect(__test.compactAsks.size).toBe(0);

    // the compaction request (the same history, extended) and the resend after it reach Cursor
    const summary = await turn([...after(`The shell returned ${TOOLLESS}.`), { role: "user", content: "Summarize the conversation." }], frame(textDelta("summary")));
    expect(summary.sse.content).toBe("summary");
    const compacted: Msg[] = [SYSTEM, { role: "user", content: "This session is being continued from a previous conversation. Summary: ..." }, { role: "user", content: "go on" }];
    const resent = await turn(compacted, frame(mcpCall(1, "call-1", "ls")));
    expect(resent.sse.finish).toBe("tool_calls");
  });

  test("a retry of the same history meets the ask too, and only once", async () => {
    __test.limits.compactTokens = 1;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta(TOOLLESS)));
    expect((await ask(FIRST)).status).toBe(400);
    const again = await turn(FIRST, frame(textDelta("noted")));
    expect(again.sse.content).toBe("noted");
  });

  test("another history is not asked", async () => {
    __test.limits.compactTokens = 1;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta(TOOLLESS)));
    const other = await turn([SYSTEM, { role: "user", content: "a subagent's task" }], frame(textDelta("done")));
    expect(other.sse.content).toBe("done");
    expect(__test.compactAsks.size).toBe(1);
  });

  test("a toolless replay under the floor is logged with its size and asks nothing", async () => {
    expect(__test.limits.compactTokens).toBe(100_000);
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta(TOOLLESS)));
    expect(__test.compactAsks.size).toBe(0);
    expect(logged.some((l) => /toolless on a fresh replay of 2 messages \([\d.]+ KB, about \d+ tokens\); under the 100000-token compaction floor \(CURSOR_BRIDGE_COMPACT_TOKENS\)/.test(l))).toBe(true);
    const two = await turn(after(TOOLLESS), frame(textDelta("ok")));
    expect(two.sse.content).toBe("ok");
  });

  test("a floor of 0 turns the ask off", async () => {
    __test.limits.compactTokens = 0;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta(TOOLLESS)));
    expect(__test.compactAsks.size).toBe(0);
    expect(logged.some((l) => l.includes("compaction asks are off (CURSOR_BRIDGE_COMPACT_TOKENS=0)"))).toBe(true);
  });

  test("a toolless continuation asks nothing: compacting is the cure for a replayed history", async () => {
    __test.limits.compactTokens = 1;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta("noted")));
    await turn(after("noted"), checkpointFrame("t2"), frame(textDelta(TOOLLESS)));
    expect(__test.compactAsks.size).toBe(0);
    expect(logged.some((l) => l.includes("no tool call of this run reached the bridge) on a continuation of a kept conversation"))).toBe(true);
  });

  test("a fresh replay whose tool calls reach the bridge asks nothing", async () => {
    __test.limits.compactTokens = 1;
    const pending = ask(FIRST);
    const fb = bridges.at(-1)!;
    fb.feed(checkpointFrame("t1"), frame(textDelta(`quoting ${TOOLLESS}`)), frame(mcpCall(1, "call-1", "ls")));
    expect((await readSse(await pending)).finish).toBe("tool_calls");
    fb.close(0);
    expect(__test.compactAsks.size).toBe(0);
  });

  test("past the idle limit an ask is dropped", async () => {
    __test.limits.compactTokens = 1;
    await turn(FIRST, checkpointFrame("t1"), frame(textDelta(TOOLLESS)));
    __test.limits.keptTtlMs = -1;
    const two = await turn(after(TOOLLESS), frame(textDelta("ok")));
    expect(two.sse.content).toBe("ok");
  });

  test("CURSOR_BRIDGE_COMPACT_TOKENS: a whole number, 0 for off, anything else the default with a warning", () => {
    const warned: string[] = [];
    const warn = (m: string) => { warned.push(m); };
    expect(__test.compactFloor(undefined, warn)).toBe(100_000);
    expect(__test.compactFloor(" ", warn)).toBe(100_000);
    expect(__test.compactFloor("120000", warn)).toBe(120_000);
    expect(__test.compactFloor("0", warn)).toBe(0);
    expect(warned).toEqual([]);
    expect(__test.compactFloor("90k", warn)).toBe(100_000);
    expect(warned[0]).toContain('CURSOR_BRIDGE_COMPACT_TOKENS="90k"');
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
