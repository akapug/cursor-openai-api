// helm arm (2026-09-25): the bridge spends the owner's Cursor account and
// takes no credential from its callers, so it listens on loopback only.
import { afterAll, describe, expect, test } from "bun:test";
import { networkInterfaces } from "node:os";
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
});
