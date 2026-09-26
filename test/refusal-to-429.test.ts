// helm arm (2026-09-16): a Connect end-stream refusal on the FIRST frame must
// resolve the peek as a refusal (so the handler answers 429), a normal first
// frame must not, and every byte seen during the peek must remain in `buf`.
import { describe, expect, test } from "bun:test";
import { __test } from "../src/proxy";

const END = 0b00000010;
function frame(flags: number, body: Uint8Array): Buffer {
  const head = Buffer.alloc(5);
  head[0] = flags;
  head.writeUInt32BE(body.length, 1);
  return Buffer.concat([head, Buffer.from(body)]);
}
function fakeBridge() {
  let onData: ((c: Buffer) => void) | null = null;
  let onClose: ((code: number) => void) | null = null;
  return {
    bridge: {
      proc: null as unknown as never,
      write: () => {},
      end: () => {},
      onData: (cb: (c: Buffer) => void) => { onData = cb; },
      onClose: (cb: (code: number) => void) => { onClose = cb; },
    },
    feed: (b: Buffer) => onData?.(b),
    close: (code: number) => onClose?.(code),
  };
}

describe("first-frame peek", () => {
  test("resource_exhausted on the first frame is a refusal", async () => {
    const f = fakeBridge();
    const p = __test.peekFirstFrame(f.bridge as never, 1000);
    const body = new TextEncoder().encode(JSON.stringify({ error: { code: "resource_exhausted", message: "Error" } }));
    f.feed(frame(END, body));
    const r = await p;
    expect(r.refusal?.code).toBe("resource_exhausted");
    expect(r.buf.length).toBe(5 + body.length);
  });
  test("a non-refusal end-stream error is not a refusal (no 429; the run's response ends with it as an error)", async () => {
    const f = fakeBridge();
    const p = __test.peekFirstFrame(f.bridge as never, 1000);
    f.feed(frame(END, new TextEncoder().encode(JSON.stringify({ error: { code: "failed_precondition", message: "x" } }))));
    expect((await p).refusal).toBeNull();
  });
  test("a normal first frame is not a refusal and split delivery keeps every byte", async () => {
    const f = fakeBridge();
    const p = __test.peekFirstFrame(f.bridge as never, 1000);
    const fr = frame(0, new Uint8Array([1, 2, 3, 4, 5, 6]));
    f.feed(fr.subarray(0, 3));
    f.feed(fr.subarray(3));
    const r = await p;
    expect(r.refusal).toBeNull();
    expect(Buffer.compare(r.buf, fr)).toBe(0);
  });
  test("a close before any frame resolves with the code recorded", async () => {
    const f = fakeBridge();
    const p = __test.peekFirstFrame(f.bridge as never, 1000);
    f.close(7);
    expect((await p).closedCode).toBe(7);
  });
  test("connectRefusal only matches the refusal codes", () => {
    expect(__test.connectRefusal(new Error("Connect error resource_exhausted: Error"))?.code).toBe("resource_exhausted");
    expect(__test.connectRefusal(new Error("Connect error unavailable: down"))?.code).toBe("unavailable");
    expect(__test.connectRefusal(new Error("Connect error failed_precondition: x"))).toBeNull();
    expect(__test.connectRefusal(new Error("something else"))).toBeNull();
  });
});
