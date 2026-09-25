// helm arm (2026-09-25): the bridge spends the owner's Cursor account and
// takes no credential from its callers, so it listens on loopback only; and
// a failed model discovery is an error, never a made-up list.
import { afterAll, describe, expect, test } from "bun:test";
import { networkInterfaces } from "node:os";
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
});
