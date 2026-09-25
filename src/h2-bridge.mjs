#!/usr/bin/env node
/**
 * Dumb HTTP/2 bidirectional pipe for Cursor gRPC.
 *
 * Bun's node:http2 is broken. This Node script acts as a transparent
 * HTTP/2 proxy: it opens a single bidirectional stream and ferries
 * raw bytes between the parent process (via stdin/stdout) and Cursor.
 *
 * Protocol (length-prefixed framing over stdin/stdout):
 *   [4 bytes big-endian length][payload]
 *
 * First message on stdin is JSON config:
 *   { "accessToken": "...", "url": "...", "path": "..." }
 *
 * After config, subsequent stdin messages are raw bytes to write to the H2 stream.
 * H2 response data is written to stdout using the same length-prefixed framing.
 */
import http2 from "node:http2";
import crypto from "node:crypto";

const CURSOR_CLIENT_VERSION = "cli-2026.01.09-231024f";

/** Write one length-prefixed message to stdout. */
function writeMessage(data) {
  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(data.length, 0);
  process.stdout.write(lenBuf);
  process.stdout.write(data);
}

// --- Buffered stdin reader ---

let stdinBuf = Buffer.alloc(0);
let stdinResolve = null;
let stdinEnded = false;

process.stdin.on("data", (chunk) => {
  stdinBuf = Buffer.concat([stdinBuf, chunk]);
  if (stdinResolve) {
    const r = stdinResolve;
    stdinResolve = null;
    r();
  }
});

process.stdin.on("end", () => {
  stdinEnded = true;
  if (stdinResolve) {
    const r = stdinResolve;
    stdinResolve = null;
    r();
  }
});

function waitForData() {
  return new Promise((resolve) => { stdinResolve = resolve; });
}

async function readExact(n) {
  while (stdinBuf.length < n) {
    if (stdinEnded) return null;
    await waitForData();
  }
  const result = stdinBuf.subarray(0, n);
  stdinBuf = stdinBuf.subarray(n);
  return Buffer.from(result);
}

async function readMessage() {
  const lenBuf = await readExact(4);
  if (!lenBuf) return null;
  const len = lenBuf.readUInt32BE(0);
  if (len === 0) return Buffer.alloc(0);
  return readExact(len);
}

// --- Main ---

const configBuf = await readMessage();
if (!configBuf) process.exit(1);

const config = JSON.parse(configBuf.toString("utf8"));
const { accessToken, url, path: rpcPath } = config;

const client = http2.connect(url || "https://api2.cursor.sh");

// helm patch (2026-09-25): NO WALL-CLOCK DEADLINE ON THE STREAM. This child
// used to kill itself 120 s after spawn whatever was happening, so any Cursor
// run longer than two minutes — a long think, or a run paused while the
// client executes a tool — was cut off and ended as a clean "stop" with no
// answer. A run's lifetime belongs to the parent, which knows whether it is
// streaming, paused or abandoned; this child ends when the parent closes
// stdin, when Cursor ends the stream, or on a transport error, and says why
// on stderr.
let closing = false;
const fail = (what, err) => {
  // Tearing the session down on purpose can still surface a stream error
  // (a GOAWAY racing the close); that is the shutdown we asked for.
  if (closing) process.exit(0);
  process.stderr.write(`[h2] ${what}: ${err?.code ?? ""} ${err?.message ?? err}\n`);
  process.exit(1);
};

client.on("error", (err) => fail("session error", err));

const h2Stream = client.request({
  ":method": "POST",
  ":path": rpcPath || "/agent.v1.AgentService/Run",
  "content-type": "application/connect+proto",
  "connect-protocol-version": "1",
  te: "trailers",
  authorization: `Bearer ${accessToken}`,
  "x-ghost-mode": "true",
  "x-cursor-client-version": CURSOR_CLIENT_VERSION,
  "x-cursor-client-type": "cli",
  "x-request-id": crypto.randomUUID(),
});

// Forward H2 response data → stdout (length-prefixed)
h2Stream.on("data", (chunk) => {
  writeMessage(chunk);
});

h2Stream.on("end", () => {
  client.close();
  // Give stdout time to flush
  setTimeout(() => process.exit(0), 100);
});

h2Stream.on("error", (err) => {
  client.close();
  fail("stream error", err);
});

// Forward stdin → H2 stream (after config message)
(async () => {
  while (true) {
    const msg = await readMessage();
    if (!msg || msg.length === 0) {
      // EOF or zero-length = the parent is done with this run. Close our half
      // and leave: before, the loop just stopped reading and the child stayed
      // up holding the Cursor stream until Cursor or the old timer ended it.
      closing = true;
      if (!h2Stream.closed && !h2Stream.destroyed) h2Stream.close();
      client.close();
      setTimeout(() => process.exit(0), 100);
      break;
    }
    if (!h2Stream.closed && !h2Stream.destroyed) {
      h2Stream.write(msg);
    }
  }
})();
