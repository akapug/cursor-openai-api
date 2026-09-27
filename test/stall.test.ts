// helm arm (2026-09-25, 2026-09-27): a client attached to a run Cursor has
// gone silent on, or that waits on a tool call Cursor never completes, is
// told so in words and the run is ended, instead of hanging until the
// client's idle timeout turns the turn into "unexpected EOF". A model that
// is thinking is none of these, heartbeats or not.
import { afterEach, describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";
import {
  fakeBridge, frame, heartbeat, makeRun, readSse, shellStreamExec, textDelta,
  toolCallCompleted, toolCallStarted, unknownExec,
} from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
const saved = { ...__test.limits };
afterEach(() => Object.assign(__test.limits, saved));

function attached() {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge);
  const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
  return { fb, resp };
}

/** Feed a heartbeat every `everyMs`, as Cursor does every 10 s, until stopped. */
function beating(fb: ReturnType<typeof fakeBridge>, everyMs = 20) {
  const t = setInterval(() => fb.feed(frame(heartbeat())), everyMs);
  return () => clearInterval(t);
}

describe("stall watchdog", () => {
  test("a model that streams thinking for longer than the stall window is not ended", async () => {
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 5_000;
    const { fb, resp } = attached();
    for (let i = 0; i < 12; i++) {
      fb.feed(frame(textDelta(`t${i} `, true)));
      await Bun.sleep(20);
    }
    fb.feed(frame(textDelta("done")));
    fb.close(0);
    const s = await readSse(resp);
    expect(s.reasoning).toBe("t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 ");
    expect(s.content).toBe("done");
    expect(s.finish).toBe("stop");
    expect(fb.state.killed).toBe(false);
  });

  test("a model thinking with heartbeats only and no tool call open is not a stall", async () => {
    // MEASURED 2026-09-26 on the cursor seat: runs that thought for minutes
    // were ended at 90 s with "made no progress (heartbeats only)". With no
    // tool call open, Cursor is not waiting on the bridge: its heartbeats say
    // the model is working.
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 5_000;
    const { fb, resp } = attached();
    fb.feed(frame(textDelta("planning", true)));
    const stop = beating(fb);
    await Bun.sleep(250);
    stop();
    fb.feed(frame(textDelta("answer")));
    fb.close(0);
    const s = await readSse(resp);
    expect(s.content).toBe("answer");
    expect(s.content).not.toContain("[Error");
    expect(fb.state.killed).toBe(false);
  });

  test("a stream that sends nothing, not even a heartbeat, is ended at the stall window", async () => {
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 5_000;
    const { fb, resp } = attached();
    const t0 = Date.now();
    fb.feed(frame(textDelta("partial answer")));
    const s = await readSse(resp);
    const took = Date.now() - t0;
    expect(s.content).toContain("partial answer");
    expect(s.content).toContain("Cursor sent nothing for 0.06 s, not even a heartbeat");
    expect(s.finish).toBe("stop");
    expect(fb.state.killed).toBe(true);
    expect(took).toBeGreaterThanOrEqual(55);
    expect(took).toBeLessThan(2_000);
  });

  // THE 09-25 STALL, as the bridge log traced it: toolCallStarted, Cursor's
  // exec, then heartbeats every 10 s and never a toolCallCompleted, because
  // the bridge's answer never reached Cursor (a refused streamed shell sent on
  // the wrong channel; an MCP state query it could not decode). Both shapes.
  for (const [shape, exec] of [
    ["a streamed shell the bridge refused", frame(shellStreamExec(4))],
    ["an exec the bridge cannot answer", unknownExec(5)],
  ] as const) {
    test(`the 09-25 stall (${shape}): a tool call left open with heartbeats only is ended at the stall window`, async () => {
      __test.limits.stallMs = 80;
      __test.limits.idleGuardMs = 5_000;
      const { fb, resp } = attached();
      const t0 = Date.now();
      fb.feed(frame(textDelta("I'll run that", true)), frame(toolCallStarted("call-1")), exec);
      const stop = beating(fb);
      const s = await readSse(resp);
      stop();
      const took = Date.now() - t0;
      expect(s.content).toContain("Cursor made no progress for 0.08 s with a tool call open (heartbeats only)");
      expect(fb.state.killed).toBe(true);
      expect(took).toBeGreaterThanOrEqual(75);
      expect(took).toBeLessThan(2_000);
    });
  }

  test("a tool call Cursor completed leaves the run thinking: heartbeats after it are not a stall", async () => {
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 5_000;
    const { fb, resp } = attached();
    fb.feed(frame(toolCallStarted("call-1")), frame(shellStreamExec(4)), frame(toolCallCompleted("call-1")));
    const stop = beating(fb);
    await Bun.sleep(250);
    stop();
    fb.feed(frame(textDelta("the refusal named Bash, so I'll use it")));
    fb.close(0);
    const s = await readSse(resp);
    expect(s.content).toBe("the refusal named Bash, so I'll use it");
    expect(fb.state.killed).toBe(false);
  });

  test("silent thinking is ended in words before the client's idle timeout would cut it", async () => {
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 200;
    const { fb, resp } = attached();
    const t0 = Date.now();
    fb.feed(frame(textDelta("thinking hard", true)));
    const stop = beating(fb);
    const s = await readSse(resp);
    stop();
    const took = Date.now() - t0;
    expect(s.content).toContain("the client got nothing for 0.2 s");
    expect(s.content).toContain(`its idle timeout is ${__test.IDLE_TIMEOUT_S} s`);
    expect(fb.state.killed).toBe(true);
    expect(took).toBeGreaterThanOrEqual(190);
  });

  test("the idle guard counts bytes the client got, not frames Cursor sent", async () => {
    // Tool bookkeeping is progress for the watchdog but reaches the client as
    // nothing, and the client's idle timeout counts only what it receives.
    __test.limits.stallMs = 5_000;
    __test.limits.idleGuardMs = 150;
    const { fb, resp } = attached();
    fb.feed(frame(textDelta("start")));
    let n = 0;
    const t = setInterval(() => {
      n++;
      fb.feed(frame(toolCallStarted(`c${n}`)), frame(toolCallCompleted(`c${n}`)));
    }, 20);
    const s = await readSse(resp);
    clearInterval(t);
    expect(s.content).toContain("the client got nothing for 0.15 s");
    expect(fb.state.killed).toBe(true);
  });

  test("a response the client keeps receiving is never cut by the idle guard", async () => {
    __test.limits.stallMs = 5_000;
    __test.limits.idleGuardMs = 100;
    const { fb, resp } = attached();
    for (let i = 0; i < 15; i++) {
      fb.feed(frame(textDelta(`${i % 10}`)));
      await Bun.sleep(20);
    }
    fb.close(0);
    const s = await readSse(resp);
    expect(s.content).toBe("012345678901234");
    expect(fb.state.killed).toBe(false);
  });

  test("a resumed run starts every clock fresh: a pause longer than both windows is not a stall", async () => {
    __test.limits.stallMs = 60;
    __test.limits.idleGuardMs = 100;
    const { fb, resp } = attached();
    const { mcpCall } = await import("./helpers");
    fb.feed(frame(toolCallStarted("c1")), frame(mcpCall(1, "call-1", "sleep 1")));
    const first = await readSse(resp);
    expect(first.finish).toBe("tool_calls");
    await Bun.sleep(200);                      // the client's tool runs; nothing is watched
    const paused = __test.pausedRuns.get(first.toolCalls[0]!.id)!;
    const next = __test.resumeRun(paused, [{ toolCallId: first.toolCalls[0]!.id, content: "out" }], "", true);
    // the first frame carries no text, so nothing replayed at attach can
    // refresh the client's clock for it
    fb.feed(frame(toolCallCompleted("c1")));
    const answer = await next!;
    await Bun.sleep(20);
    fb.feed(frame(textDelta("after")));
    fb.close(0);
    const s = await readSse(answer);
    expect(s.content).toBe("after");
    expect(fb.state.killed).toBe(false);
    __test.pausedRuns.clear();
  });

  test("steady frames keep the watchdog quiet", async () => {
    __test.limits.stallMs = 80;
    const { fb, resp } = attached();
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
    __test.limits.idleGuardMs = 40;
    const { fb, resp } = attached();
    const { mcpCall } = await import("./helpers");
    fb.feed(frame(mcpCall(1, "call-1", "sleep 1")));
    await readSse(resp);
    await Bun.sleep(120);
    expect(fb.state.killed).toBe(false);
    __test.pausedRuns.clear();
  });

  test("the first-frame wait, watchdog, and scheduling margin all fit before Bun's idle timeout", () => {
    const idleMs = __test.IDLE_TIMEOUT_S * 1000;
    expect(__test.IDLE_TIMEOUT_S).toBe(255);      // Bun.serve's maximum
    expect(saved.firstFrameMs + saved.idleGuardMs + __test.IDLE_MARGIN_MS).toBe(idleMs);
    expect(saved.stallMs).toBe(90_000);
    expect(__test.stallWindow("999999999", () => {})).toBeLessThanOrEqual(saved.idleGuardMs);
  });
});

describe("stall window from CURSOR_BRIDGE_STALL_MS", () => {
  const read = (raw: string | undefined) => {
    const warned: string[] = [];
    return { ms: __test.stallWindow(raw, (m) => warned.push(m)), warned };
  };

  test("unset or empty is the 90 s default, silently", () => {
    for (const raw of [undefined, "", "  "]) {
      expect(read(raw)).toEqual({ ms: 90_000, warned: [] });
    }
  });

  test("a whole number of milliseconds inside the bounds moves the window", () => {
    expect(read("120000")).toEqual({ ms: 120_000, warned: [] });
    expect(read(" 45000 ")).toEqual({ ms: 45_000, warned: [] });
  });

  test("a value at or past the client's idle timeout is held below it, with a warning", () => {
    const { ms, warned } = read("600000");
    expect(ms).toBe(__test.limits.idleGuardMs);
    expect(ms).toBeLessThan(__test.IDLE_TIMEOUT_S * 1000);
    expect(warned.length).toBe(1);
    expect(warned[0]).toContain("CURSOR_BRIDGE_STALL_MS=600000");
    expect(warned[0]).toContain(`idle timeout (${__test.IDLE_TIMEOUT_S} s)`);
  });

  test("a value shorter than three of Cursor's heartbeat periods is raised to that floor, with a warning", () => {
    const { ms, warned } = read("5000");
    expect(ms).toBe(30_000);
    expect(warned.length).toBe(1);
    expect(warned[0]).toContain("CURSOR_BRIDGE_STALL_MS=5000");
  });

  test("garbage is the default, with a warning that names it", () => {
    for (const raw of ["abc", "90s", "1e5", "-1", "12.5", "0x10", "Infinity", "99999999999999999999"]) {
      const { ms, warned } = read(raw);
      expect(ms).toBe(90_000);
      expect(warned.length).toBe(1);
      expect(warned[0]).toContain(`CURSOR_BRIDGE_STALL_MS=${JSON.stringify(raw)}`);
      expect(warned[0]).toContain("90000 ms default");
    }
  });

  test("the bridge reads the variable at start: set moves the window, garbage keeps the default and warns", () => {
    const src = `${import.meta.dir}/../src/proxy.ts`;
    const probe = (value: string) => Bun.spawnSync(
      [process.execPath, "-e", `import { __test } from ${JSON.stringify(src)}; console.log(__test.limits.stallMs)`],
      { env: { ...process.env, CURSOR_BRIDGE_STALL_MS: value } },
    );
    const set = probe("120000");
    expect(set.exitCode).toBe(0);
    expect(set.stdout.toString().trim()).toBe("120000");
    expect(set.stderr.toString()).not.toContain("CURSOR_BRIDGE_STALL_MS");
    const junk = probe("ninety");
    expect(junk.exitCode).toBe(0);
    expect(junk.stdout.toString().trim()).toBe("90000");
    expect(junk.stderr.toString()).toContain('CURSOR_BRIDGE_STALL_MS="ninety"');
  });
});
