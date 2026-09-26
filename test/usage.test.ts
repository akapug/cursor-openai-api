// helm arm (2026-09-25): every terminal chunk carries usage, from Cursor's
// own counts (token_delta for output, a checkpoint's used_tokens for the
// context), so Claude Code's autocompact sees the real window. A tool-call
// turn used to carry none and reached the client as 0/0 (task/2640).
//
// helm arm (2026-09-26): NO ESTIMATE IS EVER USAGE. A response that got no
// checkpoint reported ceil(promptChars / 4), the UTF-8 bytes of every
// root-prompt blob over four. MEASURED on the cursor seat: two fresh-request
// responses four minutes apart on the same conversation reported 94,263 (no
// checkpoint) and 48,828 (a checkpoint), and the resumed response between
// them 48,002; earlier one reported 397,729 where Cursor's own count was
// 79,652. Claude Code believed the guess and compacted on it. A response now
// reports Cursor's count: its own checkpoint, else the last one Cursor sent
// for the same client conversation, else no usage at all.
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import { checkpoint, fakeBridge, frame, makeRun, mcpCall, readSse, textDelta, tokenDelta } from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
afterEach(() => {
  __test.pausedRuns.clear();
  (__test as { conversationContext?: Map<string, number> }).conversationContext?.clear();
});

/** A client conversation, as Claude Code sends it, `turns` messages long. */
function conversation(opening: string, turns = 1) {
  const messages: Array<{ role: "user" | "assistant"; content: string }> = [{ role: "user", content: opening }];
  for (let i = 1; i < turns; i++) messages.push({ role: i % 2 ? "assistant" : "user", content: `turn ${i}` });
  return messages;
}

/** A run as startRun builds it: `promptChars` is what the old estimate divided by four. */
function started(opts: { stream?: boolean; opening?: string; turns?: number; promptChars?: number } = {}) {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge) as { promptChars: number };
  run.promptChars = opts.promptChars ?? 400;
  if (opts.opening) __test.stampConversation(run as never, conversation(opts.opening, opts.turns));
  const resp = __test.respond(run as never, opts.stream ?? true, () => __test.listen(run as never, NO_PEEK));
  return { fb, run, resp };
}

async function jsonUsage(resp: Response | Promise<Response>) {
  const body = await (await resp).json() as { usage?: Record<string, unknown>; choices: Array<{ finish_reason: string }> };
  return { usage: body.usage ?? null, finish: body.choices[0]!.finish_reason };
}

describe("usage", () => {
  test("a finished answer reports Cursor's counts", async () => {
    const { fb, resp } = started();
    fb.feed(frame(textDelta("thinking", true)), frame(tokenDelta(3)), frame(textDelta("answer")), frame(tokenDelta(5)), frame(checkpoint(8481)));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("stop");
    expect(s.usage).toEqual({ prompt_tokens: 8481, completion_tokens: 8, total_tokens: 8489, helm_context: "response" });
  });

  test("a tool-call turn carries usage too", async () => {
    const { fb, resp } = started();
    fb.feed(frame(checkpoint(7710)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("tool_calls");
    expect(s.usage).toEqual({ prompt_tokens: 7710, completion_tokens: 203, total_tokens: 7913, helm_context: "response" });
  });

  test("a resumed turn counts only its own output, and the context Cursor last reported", async () => {
    const { fb, run, resp } = started();
    fb.feed(frame(checkpoint(7710)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    await readSse(resp as Response);
    const resumed = __test.resumeRun(run as never, [{ toolCallId: "call-1", content: "one" }], "", true);
    fb.feed(frame(checkpoint(7900)), frame(textDelta("done")), frame(tokenDelta(16)), frame(checkpoint(7944)));
    fb.close(0);
    const s = await readSse(await resumed!);
    expect(s.usage).toEqual({ prompt_tokens: 7944, completion_tokens: 16, total_tokens: 7960, helm_context: "response" });
  });

  test("a fresh response with no checkpoint reports the conversation's last Cursor count, never the character estimate", async () => {
    // the conversation's earlier run: Cursor counted 48,002
    const a = started({ opening: "the seat's task", turns: 3 });
    a.fb.feed(frame(checkpoint(48002)), frame(tokenDelta(9)), frame(textDelta("ok")));
    a.fb.close(0);
    expect((await readSse(a.resp as Response)).usage?.prompt_tokens).toBe(48002);
    // the next user turn starts a fresh run over the same conversation, whose
    // root blobs are 377,052 bytes: the estimate would have said 94,263
    const b = started({ opening: "the seat's task", turns: 5, promptChars: 377052 });
    b.fb.feed(frame(textDelta("running it")), frame(tokenDelta(12)), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(b.resp as Response);
    expect(s.finish).toBe("tool_calls");
    expect(s.usage).toEqual({ prompt_tokens: 48002, completion_tokens: 12, total_tokens: 48014, helm_context: "conversation" });
  });

  test("a resumed response with no checkpoint carries its own conversation's count", async () => {
    const { fb, run, resp } = started({ opening: "the seat's task", turns: 3, promptChars: 600000 });
    fb.feed(frame(checkpoint(141519)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    expect((await readSse(resp as Response)).usage?.prompt_tokens).toBe(141519);
    const resumed = __test.resumeRun(run as never, [{ toolCallId: "call-1", content: "one" }], "", true);
    fb.feed(frame(textDelta("done")), frame(tokenDelta(4)));
    fb.close(0);
    const s = await readSse(await resumed!);
    expect(s.usage).toEqual({ prompt_tokens: 141519, completion_tokens: 4, total_tokens: 141523, helm_context: "conversation" });
  });

  test("a compacted conversation never carries the old conversation's count, and with no count of its own reports no usage", async () => {
    // MEASURED on the live seat: after compactions the next responses
    // reported 141,519 and then 149,317 twice, an earlier conversation's
    // count carried onto the compacted one, which kept Claude Code compacting.
    const a = started({ opening: "the seat's task", turns: 9 });
    a.fb.feed(frame(checkpoint(141519)), frame(textDelta("ok")));
    a.fb.close(0);
    await readSse(a.resp as Response);
    const b = started({ opening: "This session is being continued from a previous conversation. Summary: ...", promptChars: 90000 });
    b.fb.feed(frame(textDelta("resuming")), frame(tokenDelta(7)), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(b.resp as Response);
    expect(s.finish).toBe("tool_calls");
    expect(s.usage).toBeNull();
  });

  test("with no count from Cursor the response carries no usage, never a guess", async () => {
    const { fb, resp } = started({ opening: "a new conversation", promptChars: 400000 });
    fb.feed(frame(textDelta("12345678")));
    fb.close(0);
    const s = await readSse(resp as Response);
    expect(s.finish).toBe("stop");
    expect(s.content).toBe("12345678");
    expect(s.usage).toBeNull();
  });

  test("output is only what Cursor counted, never characters over four", async () => {
    const { fb, resp } = started();
    fb.feed(frame(checkpoint(9000)), frame(textDelta("x".repeat(4000))));
    fb.close(0);
    expect((await readSse(resp as Response)).usage).toEqual({ prompt_tokens: 9000, completion_tokens: 0, total_tokens: 9000, helm_context: "response" });
  });

  test("a checkpoint in the response wins over the conversation's carried count", async () => {
    const a = started({ opening: "the seat's task" });
    a.fb.feed(frame(checkpoint(60000)), frame(textDelta("ok")));
    a.fb.close(0);
    await readSse(a.resp as Response);
    const b = started({ opening: "the seat's task", turns: 3 });
    b.fb.feed(frame(textDelta("ok")), frame(checkpoint(48002)));
    b.fb.close(0);
    expect((await readSse(b.resp as Response)).usage?.prompt_tokens).toBe(48002);
  });
});

describe("usage, non-streamed", () => {
  test("a checkpoint is reported as Cursor counted it", async () => {
    const { fb, resp } = started({ stream: false });
    fb.feed(frame(textDelta("answer")), frame(tokenDelta(5)), frame(checkpoint(8481)));
    fb.close(0);
    expect((await jsonUsage(resp)).usage).toEqual({ prompt_tokens: 8481, completion_tokens: 5, total_tokens: 8486, helm_context: "response" });
  });

  test("no checkpoint carries the conversation's count, and none at all omits usage", async () => {
    const a = started({ stream: false, opening: "the seat's task" });
    a.fb.feed(frame(checkpoint(48002)), frame(textDelta("ok")));
    a.fb.close(0);
    await jsonUsage(a.resp);
    const b = started({ stream: false, opening: "the seat's task", turns: 3, promptChars: 377052 });
    b.fb.feed(frame(textDelta("ok")), frame(tokenDelta(2)));
    b.fb.close(0);
    expect((await jsonUsage(b.resp)).usage).toEqual({ prompt_tokens: 48002, completion_tokens: 2, total_tokens: 48004, helm_context: "conversation" });
    const c = started({ stream: false, opening: "another conversation", promptChars: 377052 });
    c.fb.feed(frame(textDelta("ok")));
    c.fb.close(0);
    const got = await jsonUsage(c.resp);
    expect(got.finish).toBe("stop");
    expect(got.usage).toBeNull();
  });
});
