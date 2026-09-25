// helm arm (2026-09-25): the h2 child's lifetime belongs to the parent. It
// used to kill itself 120 s after spawn (cutting every longer Cursor run to
// an empty "stop"), and on stdin EOF it merely stopped reading, holding the
// Cursor stream open. It must now leave promptly on EOF, closing its half.
import { afterEach, describe, expect, test } from "bun:test";
import { resolve } from "node:path";

const CHILD = resolve(import.meta.dir, "../src/h2-bridge.mjs");
const SERVER = resolve(import.meta.dir, "fixtures/h2c-server.mjs");
const procs: Array<{ kill: () => void }> = [];
afterEach(() => { for (const p of procs.splice(0)) p.kill(); });

function lp(data: Uint8Array): Uint8Array {
  const b = Buffer.alloc(4 + data.length);
  b.writeUInt32BE(data.length, 0);
  b.set(data, 4);
  return b;
}

async function lines(stream: ReadableStream<Uint8Array>, want: (l: string) => boolean, ms: number): Promise<string[]> {
  const seen: string[] = [];
  const reader = stream.getReader();
  const deadline = Date.now() + ms;
  let buf = "";
  while (Date.now() < deadline) {
    const r = await Promise.race([reader.read(), Bun.sleep(deadline - Date.now()).then(() => null)]);
    if (!r || r.done) break;
    buf += new TextDecoder().decode(r.value);
    const parts = buf.split("\n");
    buf = parts.pop()!;
    seen.push(...parts);
    if (parts.some(want)) break;
  }
  reader.releaseLock();
  return seen;
}

describe("h2 child lifetime", () => {
  test("stdin EOF closes the Cursor stream and exits promptly", async () => {
    const server = Bun.spawn(["node", SERVER], { stdout: "pipe" });
    procs.push(server);
    const [portLine] = await lines(server.stdout, (l) => l.startsWith("port "), 5_000);
    const port = Number(portLine!.split(" ")[1]);

    const child = Bun.spawn(["node", CHILD], { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
    procs.push(child);
    child.stdin.write(lp(new TextEncoder().encode(JSON.stringify({
      accessToken: "test", url: `http://127.0.0.1:${port}`, path: "/x", clientVersion: "test",
    }))));
    child.stdin.write(lp(new Uint8Array([0, 0, 0, 0, 0])));
    await child.stdin.flush();
    await Bun.sleep(500);
    expect(child.exitCode).toBeNull(); // alive while the parent holds stdin

    const t0 = Date.now();
    child.stdin.write(lp(new Uint8Array(0)));
    child.stdin.end();
    const code = await Promise.race([child.exited, Bun.sleep(3_000).then(() => "still running")]);
    expect(code).toBe(0);
    expect(Date.now() - t0).toBeLessThan(3_000);
    const events = await lines(server.stdout, (l) => l === "stream-closed", 2_000);
    expect(events).toContain("stream-closed");
  });
});
