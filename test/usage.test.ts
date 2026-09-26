// helm arm (2026-09-25): every terminal chunk carries usage, from Cursor's
// own counts (token_delta for output, a checkpoint's used_tokens for the
// context), so Claude Code's autocompact sees the real window. A tool-call
// turn used to carry none and reached the client as 0/0 (task/2640).
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import { checkpoint, fakeBridge, frame, makeRun, mcpCall, readSse, textDelta, tokenDelta } from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
afterEach(() => __test.pausedRuns.clear());

function started() {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge);
  const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
  return { fb, run, resp };
}

describe("usage", () => {
  test("a finished answer reports Cursor's counts, not the estimate", async () => {
    const { fb, resp } = started();
    fb.feed(frame(textDelta("thinking", true)), frame(tokenDelta(3)), frame(textDelta("answer")), frame(tokenDelta(5)), frame(checkpoint(8481)));
    fb.close(0);
    const s = await readSse(resp);
    expect(s.finish).toBe("stop");
    expect(s.usage).toEqual({ prompt_tokens: 8481, completion_tokens: 8, total_tokens: 8489, helm_estimated: false });
  });

  test("a tool-call turn carries usage too", async () => {
    const { fb, resp } = started();
    fb.feed(frame(checkpoint(7710)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    const s = await readSse(resp);
    expect(s.finish).toBe("tool_calls");
    expect(s.usage).toEqual({ prompt_tokens: 7710, completion_tokens: 203, total_tokens: 7913, helm_estimated: false });
  });

  test("a resumed turn counts only its own output, and the context Cursor last reported", async () => {
    const { fb, run, resp } = started();
    fb.feed(frame(checkpoint(7710)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    await readSse(resp);
    const resumed = __test.resumeRun(run, [{ toolCallId: "call-1", content: "one" }], "", true);
    fb.feed(frame(checkpoint(7900)), frame(textDelta("done")), frame(tokenDelta(16)), frame(checkpoint(7944)));
    fb.close(0);
    const s = await readSse(await resumed!);
    expect(s.usage).toEqual({ prompt_tokens: 7944, completion_tokens: 16, total_tokens: 7960, helm_estimated: false });
  });

  test("a resumed turn with no checkpoint of its own never reports the previous response's context", async () => {
    // MEASURED on the live seat: after a compaction the next responses
    // reported 141,519 and then 149,317 twice, the last checkpoint of an
    // earlier response, carried onto a response that received none.
    const { fb, run, resp } = started();
    fb.feed(frame(checkpoint(141519)), frame(tokenDelta(203)), frame(mcpCall(1, "call-1", "echo 1")));
    expect((await readSse(resp)).usage?.prompt_tokens).toBe(141519);
    const resumed = __test.resumeRun(run, [{ toolCallId: "call-1", content: "one" }], "", true);
    fb.feed(frame(textDelta("done")), frame(tokenDelta(4)));
    fb.close(0);
    const s = await readSse(await resumed!);
    expect(s.usage).toEqual({ prompt_tokens: Math.ceil((400 + "one".length) / 4), completion_tokens: 4,
      total_tokens: Math.ceil((400 + "one".length) / 4) + 4, helm_estimated: true });
  });

  test("with no counts from Cursor the estimate is used and says so", async () => {
    const { fb, resp } = started();
    fb.feed(frame(textDelta("12345678")));
    fb.close(0);
    const s = await readSse(resp);
    expect(s.usage).toEqual({ prompt_tokens: 100, completion_tokens: 2, total_tokens: 102, helm_estimated: true });
  });
});
