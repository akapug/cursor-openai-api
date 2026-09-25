// helm arm (2026-09-25): the bridge spends the owner's Cursor account and
// takes no credential from its callers, so it listens on loopback only; a
// failed model discovery is an error, never a made-up list; and the h2 child
// refuses to start without the one client-version string the parent owns.
import { afterAll, describe, expect, test } from "bun:test";
import { networkInterfaces } from "node:os";
import { resolve } from "node:path";
import { getCursorModels } from "../src/models";
import { startProxy, stopProxy } from "../src/proxy";

afterAll(() => stopProxy());

describe("bridge safety", () => {
  test("the server answers on 127.0.0.1 and not on the machine's LAN address", async () => {
    const port = await startProxy(async () => "unused", 0);
    const local = await fetch(`http://127.0.0.1:${port}/nope`);
    expect(local.status).toBe(404);
    const lan = Object.values(networkInterfaces()).flat()
      .find((i) => i && i.family === "IPv4" && !i.internal)?.address;
    if (!lan) return; // no non-loopback interface on this machine: nothing to prove
    const reached = await fetch(`http://${lan}:${port}/nope`, { signal: AbortSignal.timeout(2_000) })
      .then(() => true, () => false);
    expect(reached).toBe(false);
  });

  test("a failed model discovery throws instead of answering a hardcoded list", async () => {
    await expect(getCursorModels("k", async () => null)).rejects.toThrow(/discovery failed/);
    await expect(getCursorModels("k", async () => [])).rejects.toThrow(/discovery failed/);
    const real = [{ id: "grok-4.7-high", name: "Grok", reasoning: true, contextWindow: 1, maxTokens: 1 }];
    expect(await getCursorModels("k", async () => real)).toEqual(real);
  });

  test("the h2 child will not start without the parent's client version", async () => {
    const child = Bun.spawn(["node", resolve(import.meta.dir, "../src/h2-bridge.mjs")], {
      stdin: "pipe", stdout: "pipe", stderr: "pipe",
    });
    const cfg = new TextEncoder().encode(JSON.stringify({ accessToken: "t", url: "http://127.0.0.1:9", path: "/x" }));
    const b = Buffer.alloc(4 + cfg.length);
    b.writeUInt32BE(cfg.length, 0);
    b.set(cfg, 4);
    child.stdin.write(b);
    await child.stdin.flush();
    expect(await child.exited).toBe(2);
    expect(await new Response(child.stderr).text()).toContain("no clientVersion");
  });
});
