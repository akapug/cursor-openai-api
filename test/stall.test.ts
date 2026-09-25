// helm arm (2026-09-25): a client attached to a run Cursor has gone silent on
// is told so in words and the run is ended, instead of hanging until an idle
// timeout turns the turn into an empty 500.
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import { fakeBridge, frame, heartbeat, makeRun, readSse, textDelta } from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
const saved = { ...__test.limits };
afterEach(() => Object.assign(__test.limits, saved));

describe("stall watchdog", () => {
  test("silence while a client waits ends the run with an error the client can read", async () => {
    __test.limits.stallMs = 60;
    const fb = fakeBridge();
    const run = makeRun(fb.bridge);
    const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
    fb.feed(frame(textDelta("partial answer")));
    const s = await readSse(resp);
    expect(s.content).toContain("partial answer");
    expect(s.content).toContain("Cursor made no progress");
    expect(s.finish).toBe("stop");
    expect(fb.state.killed).toBe(true);
  });

  test("heartbeats alone do not keep a stuck run alive", async () => {
    __test.limits.stallMs = 80;
    const fb = fakeBridge();
    const run = makeRun(fb.bridge);
    const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
    fb.feed(frame(textDelta("waiting on an exec")));
    const beats = setInterval(() => fb.feed(frame(heartbeat())), 20);
    const s = await readSse(resp);
    clearInterval(beats);
    expect(s.content).toContain("Cursor made no progress");
    expect(fb.state.killed).toBe(true);
  });

  test("steady frames keep the watchdog quiet", async () => {
    __test.limits.stallMs = 80;
    const fb = fakeBridge();
    const run = makeRun(fb.bridge);
    const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
    for (let i = 0; i < 5; i++) {
      fb.feed(frame(textDelta(`${i}`)));
      await Bun.sleep(40);
    }
    fb.close(0);
    const s = await readSse(resp);
    expect(s.content).toBe("01234");
    expect(fb.state.killed).toBe(false);
  });

  test("a paused run is not watched: silence there is the client's tool running", async () => {
    __test.limits.stallMs = 40;
    const fb = fakeBridge();
    const run = makeRun(fb.bridge);
    const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
    const { mcpCall } = await import("./helpers");
    fb.feed(frame(mcpCall(1, "call-1", "sleep 1")));
    await readSse(resp);
    await Bun.sleep(120);
    expect(fb.state.killed).toBe(false);
    __test.pausedRuns.clear();
  });
});
