// helm arm (2026-09-25): the bridge spends the owner's Cursor account and
// takes no credential from its callers, so it listens on loopback only; a
// failed model discovery is an error, never a made-up list; and the h2 child
// refuses to start without the one client-version string the parent owns.
import { afterAll, describe, expect, test } from "bun:test";
import { chmodSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { networkInterfaces, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { curlHeaderConfig, fetchCursorUsableModels, getCursorModels } from "../src/models";
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

  test("model discovery never puts the Cursor token on curl's command line", async () => {
    // A stand-in curl first on PATH records its argv and its stdin, then answers
    // 200 with an empty body. /proc/<pid>/cmdline is readable by every local
    // user, so a bearer there is a bearer anyone on the box can copy. The
    // discovery runs in a child bun whose PATH is set at its start, the way the
    // supervisor starts the bridge.
    const dir = mkdtempSync(join(tmpdir(), "fake-curl-"));
    const fake = join(dir, "curl");
    writeFileSync(fake, [
      "#!/bin/sh",
      `printf '%s\\n' "$@" > "${dir}/argv"`,
      `cat > "${dir}/stdin"`,
      'out=""; prev=""; for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done',
      '[ -n "$out" ] && : > "$out"',
      "printf 200",
      "",
    ].join("\n"));
    chmodSync(fake, 0o755);
    const secret = "SECRET-TOKEN-MARKER-7f3c";
    const models = resolve(import.meta.dir, "../src/models.ts");
    const child = Bun.spawn([process.execPath, "-e",
      `const m = await import(${JSON.stringify(models)});` +
      `await m.fetchCursorUsableModels({ apiKey: ${JSON.stringify(secret)}, ` +
      `baseUrl: "http://127.0.0.1:9", timeoutMs: 2000 });`], {
      env: { ...process.env, PATH: `${dir}:${process.env.PATH}` },
      stdout: "pipe", stderr: "pipe",
    });
    expect(await child.exited).toBe(0);
    const argv = readFileSync(join(dir, "argv"), "utf8");
    expect(argv).toContain("--http2");                 // the fake WAS the curl that ran
    expect(argv).not.toContain(secret);
    expect(readFileSync(join(dir, "stdin"), "utf8")).toContain(`Bearer ${secret}`);
  });

  test("the curl header config escapes what curl's config syntax reads", () => {
    expect(curlHeaderConfig({ a: 'x"y\\z', b: "plain" }))
      .toBe('header = "a: x\\"y\\\\z"\nheader = "b: plain"\n');
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
