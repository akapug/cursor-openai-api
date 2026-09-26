// helm arm (2026-09-26): a Connect error Cursor sends while a client waits is
// an ERROR the client retries, never reply text. MEASURED on the cursor seat:
// the request that carried the owner's tool results resumed a paused run, the
// bridge had already answered 200, and Cursor's `unavailable` reached Claude
// Code as the reply "[Error: Connect error unavailable: Error]", which ended
// the turn. The fresh-request path already turned a first-frame refusal into
// a 429; the resume path now waits for the run's first frame the same way,
// and an error after the stream began ends it with a bare JSON error line,
// which CLIProxyAPI turns into an Anthropic `event: error`.
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import { fakeBridge, frame, heartbeat, makeRun, mcpCall, readSse, textDelta } from "./helpers";

const END = 0b00000010;
const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
const saved = { ...__test.limits };
afterEach(() => {
  Object.assign(__test.limits, saved);
  __test.pausedRuns.clear();
});

function endStream(code: string, message = "Error"): Buffer {
  const body = Buffer.from(JSON.stringify({ error: { code, message } }));
  const head = Buffer.alloc(5);
  head[0] = END;
  head.writeUInt32BE(body.length, 1);
  return Buffer.concat([head, body]);
}

function started(stream = true) {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge);
  const resp = __test.respond(run, stream, () => __test.listen(run, NO_PEEK));
  return { fb, run, resp };
}

/** A run paused on one tool call, as the client sees it after the first response. */
async function paused() {
  const s = started();
  s.fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
  expect((await readSse(s.resp as Response)).finish).toBe("tool_calls");
  return s;
}

function resume(run: unknown, stream = true) {
  return __test.resumeRun(run as never, [{ toolCallId: "call-1", content: "one" }], "", stream);
}

/** Every SSE line the bridge wrote, in order. */
async function lines(resp: Response): Promise<string[]> {
  return (await resp.text()).split("\n").filter((l) => l.length > 0);
}

describe("resume path: the first frame decides the status", () => {
  for (const code of ["unavailable", "resource_exhausted"]) {
    test(`${code} as the resumed run's first frame is a 429 with Retry-After, never reply text`, async () => {
      const { fb, run } = await paused();
      const pending = resume(run);
      expect(pending).not.toBeNull();
      fb.feed(endStream(code));
      const resp = await pending!;
      expect(resp.status).toBe(429);
      expect(resp.headers.get("Retry-After")).toBe("60");
      const body = await resp.json() as { error: { code: string; type: string } };
      expect(body.error.code).toBe(code);
      expect(body.error.type).toBe("rate_limit_error");
      // the run is over: nothing holds it, and its bridge is ended
      expect(fb.state.killed).toBe(true);
      expect(__test.pausedRuns.size).toBe(0);
    });
  }

  test("heartbeats do not commit the resumed response; the refusal after them is still a 429", async () => {
    const { fb, run } = await paused();
    const pending = resume(run);
    fb.feed(frame(heartbeat()));
    fb.feed(frame(heartbeat()));
    fb.feed(endStream("unavailable"));
    expect((await pending!).status).toBe(429);
  });

  test("a normal first frame streams the resumed answer as before", async () => {
    const { fb, run } = await paused();
    const pending = resume(run);
    fb.feed(frame(textDelta("saw one")));
    fb.close(0);
    const resp = await pending!;
    expect(resp.status).toBe(200);
    const s = await readSse(resp);
    expect(s.content).toBe("saw one");
    expect(s.finish).toBe("stop");
  });

  test("a resumed run that says nothing is answered after the first-frame wait, not held forever", async () => {
    __test.limits.firstFrameMs = 30;
    const { fb, run } = await paused();
    const pending = resume(run);
    const resp = await pending!;
    expect(resp.status).toBe(200);
    fb.feed(frame(textDelta("late")));
    fb.close(0);
    expect((await readSse(resp)).content).toBe("late");
  });

  test("a non-streaming resume answers a refusal with the same 429", async () => {
    const { fb, run } = await paused();
    const pending = resume(run, false);
    fb.feed(endStream("resource_exhausted"));
    expect((await pending!).status).toBe(429);
  });
});

describe("a Connect error after the stream began", () => {
  test("ends the stream with a retryable error line, never reply text", async () => {
    const { fb, resp } = started();
    fb.feed(frame(textDelta("partial answer")));
    fb.feed(endStream("unavailable"));
    const got = await lines(resp as Response);
    const text = got.join("\n");
    expect(text).not.toContain("[Error");
    // no finish chunk and no [DONE]: the client must not read a finished turn
    expect(text).not.toContain("finish_reason\":\"stop");
    expect(got).not.toContain("data: [DONE]");
    // the last line is a bare JSON error (no `data:`), which CLIProxyAPI's
    // OpenAI-compatible reader turns into a stream error carrying its type
    const last = got.at(-1)!;
    expect(last.startsWith("{")).toBe(true);
    const err = JSON.parse(last) as { error: { type: string; code: string; message: string } };
    expect(err.error.type).toBe("overloaded_error");
    expect(err.error.code).toBe("unavailable");
    expect(err.error.message).toContain("Connect error unavailable");
    expect(fb.state.killed).toBe(true);
  });

  test("a non-refusal Connect error ends the stream with an api_error line, not text", async () => {
    const { fb, resp } = started();
    fb.feed(frame(textDelta("partial")));
    fb.feed(endStream("failed_precondition", "bad state"));
    const got = await lines(resp as Response);
    expect(got.join("\n")).not.toContain("[Error");
    const err = JSON.parse(got.at(-1)!) as { error: { type: string } };
    expect(err.error.type).toBe("api_error");
  });

  test("the same error in a resumed stream, after its first frame, ends it the same way", async () => {
    const { fb, run } = await paused();
    const pending = resume(run);
    fb.feed(frame(textDelta("working")));
    const resp = await pending!;
    fb.feed(endStream("resource_exhausted"));
    const got = await lines(resp);
    expect(got.join("\n")).not.toContain("[Error");
    expect((JSON.parse(got.at(-1)!) as { error: { type: string } }).error.type).toBe("overloaded_error");
  });

  test("a non-streaming answer that meets a refusal is an HTTP 429, not a 200 answer", async () => {
    const { fb, resp } = started(false);
    fb.feed(frame(textDelta("partial")));
    fb.feed(endStream("unavailable"));
    const r = await resp;
    expect(r.status).toBe(429);
    expect(r.headers.get("Retry-After")).toBe("60");
  });
});
