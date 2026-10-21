// Regression test (2026-09-25): every terminal chunk carries usage, so Claude Code's
// autocompact sees a count on every turn. A tool-call turn used to carry none
// and reached the client as 0/0.
//
// Regression test (2026-09-27): USAGE DESCRIBES THE REQUEST, NEVER CURSOR'S COUNT.
// Cursor's checkpoint count is the size of Cursor's own conversation, and
// it is not the conversation Claude Code sent. MEASURED on a live session's
// transcripts, 2026-09-26 and 27 (UTC):
//  - one conversation, two fresh runs 87 minutes apart, 796 bytes of message
//    text between them: Cursor counted 77,400 and then 911,810, 3.6 times its
//    own max_tokens of 256,000. The client's context meter read 474.9% of the
//    192,000 window and Claude Code's status line read "0% until
//    auto-compact". Five readings
//    over 256,000 in the 21 hours that build ran (603,845, 367,162, 982,009,
//    609,634, 911,810), each followed by a compaction within four minutes.
//  - the first response of a fresh conversation reported about 50,500 output
//    tokens (50,474 to 50,909, ten conversations) for a few hundred visible
//    tokens, in under 20 s, and the next checkpoint rose about 47,000
//    (16,310 to 63,453) while the conversation grew 3.9 KB. Claude Code adds
//    output to input when it sizes its window. While it worked, the session
//    compacted every 8 to 19 minutes.
// A response now reports an estimate of what the request carried (its
// messages and tools, UTF-8 bytes over four) and of what the response carried
// (text, reasoning and tool calls, the same way). Cursor's counts reach
// neither.
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import { checkpoint, fakeBridge, frame, makeRun, mcpCall, readSse, textDelta, tokenDelta } from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
const saved = { ...__test.limits };
afterEach(() => {
  Object.assign(__test.limits, saved);
  __test.pausedRuns.clear();
});

type Msg = { role: "system" | "user" | "assistant" | "tool"; content: unknown; tool_call_id?: string; tool_calls?: unknown[] };
type Tool = { type: "function"; function: { name: string; description?: string; parameters?: Record<string, unknown> } };

const TOOLS: Tool[] = [{
  type: "function",
  function: {
    name: "Bash",
    description: "Run a shell command.",
    parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
  },
}];

/** A request's estimate, as handleChatCompletion hands it to the run. */
function estimate(messages: Msg[], tools: Tool[] = TOOLS, stateless = true): number {
  return __test.requestTokens(messages as never, tools as never, stateless);
}

const bytes = (s: string) => Buffer.byteLength(s, "utf8");

/** The UTF-8 bytes a client received: text, reasoning, and each tool call's id, name and arguments. */
function carried(s: { content: string; reasoning: string; toolCalls: Array<{ id: string; name: string; arguments: string }> }): number {
  return bytes(s.content) + bytes(s.reasoning)
    + s.toolCalls.reduce((n, c) => n + bytes(c.id) + bytes(c.name) + bytes(c.arguments), 0);
}

/** A fresh conversation after /clear: the system prompt and one short user message. */
function fresh(text = "check your inbox and carry on"): Msg[] {
  return [{ role: "system", content: "You are a coding agent. Be brief." }, { role: "user", content: text }];
}

/** A run answering `messages`, as startRun builds it. */
function started(messages: Msg[], stream = true) {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge) as { promptTokens: number };
  run.promptTokens = estimate(messages);
  const resp = __test.respond(run as never, stream, () => __test.listen(run as never, NO_PEEK));
  return { fb, run, resp };
}

async function jsonBody(resp: Response | Promise<Response>) {
  return await (await resp).json() as {
    usage?: Record<string, unknown>;
    choices: Array<{ finish_reason: string; message: { content: string; reasoning_content?: string; tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }> } }>;
  };
}

describe("the request estimate", () => {
  test("counts every message's text, every tool call, and every tool the request carried, in UTF-8 bytes over four", () => {
    const messages: Msg[] = [
      { role: "system", content: "abcd" },                                              // 4
      { role: "user", content: [{ type: "text", text: "12345678" }] },                  // 8
      { role: "assistant", content: "ok", tool_calls: [                                 // 2
        { id: "c1", type: "function", function: { name: "Bash", arguments: "{\"command\":\"ls\"}" } }, // 4 + 16
      ] },
      { role: "tool", tool_call_id: "c1", content: [{ type: "tool_result", content: "a.txt" }] }, // 5
      { role: "user", content: "漢字" },                                                 // 6: three bytes a character
    ];
    const tools: Tool[] = [{ type: "function", function: { name: "Bash", description: "Run.", parameters: { type: "object" } } }];
    // call/result ids: 2 + 2; tool: "Bash" 4 + "Run." 4 + '{"type":"object"}' 17
    expect(__test.requestTokens(messages as never, tools as never)).toBe(Math.ceil((4 + 8 + 2 + 22 + 7 + 6 + 25) / 4));
    expect(__test.requestTokens(messages as never, [] as never)).toBe(Math.ceil((4 + 8 + 2 + 22 + 7 + 6) / 4));
  });

  test("a tool with no description or parameters counts the schema actually forwarded", () => {
    const messages: Msg[] = [{ role: "system", content: "s" }, { role: "user", content: "u" }];
    const schema = JSON.stringify({ type: "object", properties: {}, required: [] });
    expect(__test.requestTokens(messages as never, [{ type: "function", function: { name: "Ping" } }] as never))
      .toBe(Math.ceil((2 + 4 + bytes(schema)) / 4));
  });

  test("a text-empty user request counts the prompts the bridge synthesizes", () => {
    const messages: Msg[] = [{ role: "user", content: "" }];
    expect(__test.requestTokens(messages as never, [] as never))
      .toBe(Math.ceil((bytes("You are a helpful assistant.") + bytes("Continue from where you left off, using the tool results above.")) / 4));
  });
});

describe("usage, streamed", () => {
  test("a fresh small conversation reports its own size, far below the window, whatever Cursor counted", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages);
    // the measured first response: ~50,900 output tokens and a count of
    // 16,310 for a conversation of a few hundred bytes
    fb.feed(frame(checkpoint(16310)), frame(textDelta("on it", true)), frame(tokenDelta(50900)), frame(textDelta("Checking.")));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("stop");
    expect(s.usage).toEqual({
      prompt_tokens: estimate(messages),
      completion_tokens: Math.ceil(carried(s) / 4),
      total_tokens: estimate(messages) + Math.ceil(carried(s) / 4),
      usage_source: "estimate",
    });
    expect(s.usage!.prompt_tokens as number).toBeLessThan(1000);
  });

  test("Cursor's cumulative or server-side count never reaches the reported input", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages);
    // the measured runaway: 911,810 on a conversation Cursor counted 77,400 an hour before
    fb.feed(frame(checkpoint(77400)), frame(checkpoint(911810)), frame(textDelta("done")), frame(checkpoint(982009)));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.usage?.prompt_tokens).toBe(estimate(messages));
  });

  test("with no count from Cursor the response still reports the request's size, never nothing", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages);
    fb.feed(frame(textDelta("12345678")));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("stop");
    expect(s.usage).toEqual({ prompt_tokens: estimate(messages), completion_tokens: 2, total_tokens: estimate(messages) + 2, usage_source: "estimate" });
  });

  test("a tool-call turn carries usage, and its output is the tool call it carried", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages);
    fb.feed(frame(checkpoint(7710)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("tool_calls");
    expect(s.toolCalls).toHaveLength(1);
    expect(s.usage).toEqual({
      prompt_tokens: estimate(messages),
      completion_tokens: Math.ceil(carried(s) / 4),
      total_tokens: estimate(messages) + Math.ceil(carried(s) / 4),
      usage_source: "estimate",
    });
  });

  test("output is what the response carried, never Cursor's token_delta", async () => {
    const { fb, resp } = started(fresh());
    fb.feed(frame(textDelta("y".repeat(400), true)), frame(tokenDelta(50474)), frame(textDelta("x".repeat(4000))), frame(tokenDelta(9)));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.usage?.completion_tokens).toBe(Math.ceil(4400 / 4));
  });

  test("a resumed run's count describes the conversation its request carries, and grows with it", async () => {
    const first = fresh();
    const { fb, run, resp } = started(first);
    fb.feed(frame(checkpoint(16310)), frame(mcpCall(1, "call-1", "echo 1")));
    const one = await readSse(resp as Response);
    expect(one.usage?.prompt_tokens).toBe(estimate(first));
    // the client's next request: the whole conversation, now with the call and a long result
    const second: Msg[] = [
      ...first,
      { role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function", function: { name: "Bash", arguments: "{\"command\":\"echo 1\"}" } }] },
      { role: "tool", tool_call_id: "call-1", content: "r".repeat(4000) },
    ];
    const resumedTokens = estimate(second, TOOLS, false);
    const resumed = __test.resumeRun(run as never, [{ toolCallId: "call-1", content: "r".repeat(4000) }], "", true, resumedTokens);
    // Cursor's measured one-time rise after the first tool round trip
    fb.feed(frame(checkpoint(63453)), frame(textDelta("done")));
    fb.close(0);
    const two = await readSse(await resumed!);
    expect(two.usage?.prompt_tokens).toBe(resumedTokens);
    // It grew by the call, both copies of its id, and its result. A paused-run
    // resume sends those results directly, so no stateless continuation prompt
    // is counted; Cursor's rise of 47,143 is absent.
    expect((two.usage!.prompt_tokens as number) - (one.usage!.prompt_tokens as number))
      .toBe(resumedTokens - estimate(first));
  });

  test("a compacted conversation reports its own small size, never the long one's", async () => {
    const long: Msg[] = [...fresh("the session's task"), { role: "assistant", content: "z".repeat(400_000) }, { role: "user", content: "go on" }];
    const a = started(long);
    a.fb.feed(frame(checkpoint(141519)), frame(textDelta("ok")));
    a.fb.close(0);
    expect((await readSse(a.resp as Response)).usage?.prompt_tokens).toBe(estimate(long));
    const compacted = fresh("This session is being continued from a previous conversation. Summary: ...");
    const b = started(compacted);
    b.fb.feed(frame(textDelta("resuming")), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(b.resp as Response);
    expect(s.finish).toBe("tool_calls");
    expect(s.usage?.prompt_tokens).toBe(estimate(compacted));
    expect(s.usage!.prompt_tokens as number).toBeLessThan(1000);
  });
});

describe("usage, through the request handler", () => {
  test("the request that resumes a paused run sets the count from its own messages and tools", async () => {
    const first = fresh();
    const { fb, resp } = started(first);
    fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
    expect((await readSse(resp as Response)).finish).toBe("tool_calls");
    const body = {
      model: "grok-4.7-high",
      stream: true,
      tools: TOOLS,
      messages: [
        ...first,
        { role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function", function: { name: "Bash", arguments: "{\"command\":\"echo 1\"}" } }] },
        { role: "tool", tool_call_id: "call-1", content: "1\n".repeat(3000) },
      ] as Msg[],
    };
    const pending = __test.handleChatCompletion(body as never, "unused: a resumed run needs no token");
    fb.feed(frame(textDelta("printed 1")));
    fb.close(0);
    const s = await readSse(await pending);
    expect(s.content).toBe("printed 1");
    expect(s.usage?.prompt_tokens).toBe(estimate(body.messages, TOOLS, false));
    expect(s.usage?.prompt_tokens).toBeGreaterThan(estimate(first));
  });
});

describe("usage, non-streamed", () => {
  test("the estimate rides the whole answer, and Cursor's counts reach neither side", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages, false);
    fb.feed(frame(checkpoint(911810)), frame(textDelta("why", true)), frame(textDelta("answer")), frame(tokenDelta(50900)));
    fb.close(0);
    const body = await jsonBody(resp);
    const m = body.choices[0]!.message;
    const out = Math.ceil((bytes(m.content) + bytes(m.reasoning_content ?? "")) / 4);
    expect(body.choices[0]!.finish_reason).toBe("stop");
    expect(body.usage).toEqual({ prompt_tokens: estimate(messages), completion_tokens: out, total_tokens: estimate(messages) + out, usage_source: "estimate" });
  });

  test("a tool-call answer counts its tool calls as output", async () => {
    const messages = fresh();
    const { fb, resp } = started(messages, false);
    fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
    const body = await jsonBody(resp);
    const call = body.choices[0]!.message.tool_calls![0]!;
    expect(body.choices[0]!.finish_reason).toBe("tool_calls");
    expect(body.usage?.completion_tokens).toBe(Math.ceil((bytes(call.id) + bytes(call.function.name) + bytes(call.function.arguments)) / 4));
  });
});
