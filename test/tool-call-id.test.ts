// Regression test (2026-09-25): a Cursor tool-call id carries "\n"; CLIProxyAPI
// rewrites it to "_" before Claude Code sees it, and Claude Code sends the
// rewritten id back. The bridge must hand out ids already in [A-Za-z0-9_-]
// and match a returned id through the same function, so the tool RESULT
// reaches Cursor instead of "Tool result not provided".
import { describe, expect, test } from "bun:test";
import { fromBinary } from "@bufbuild/protobuf";
import { AgentClientMessageSchema, type McpResult } from "../src/proto/agent_pb";
import { __test } from "../src/proxy";
import { fakeBridge, makeRun } from "./helpers";

const RAW = "call-a1bf6871-8e90-4716-bc98-046e2fedaac5-1\nfc_92ed79e9-7606-9371-8a79-e7b663e8db4c_0";
// what CLIProxyAPI's SanitizeClaudeToolID makes of RAW
const PROXY_REWRITE = RAW.replace(/[^a-zA-Z0-9_-]/g, "_");

function mcpResultsOf(writes: Uint8Array[]): McpResult[] {
  return writes.map((w) => {
    const msg = fromBinary(AgentClientMessageSchema, w.subarray(5));
    const exec = msg.message.value as { message: { case: string; value: McpResult } };
    expect(exec.message.case).toBe("mcpResult");
    return exec.message.value;
  });
}

function resumeWith(toolCallIdSentBack: string) {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge) as { awaiting: unknown[] };
  run.awaiting.push({
    execId: "e1",
    execMsgId: 3,
    toolCallId: __test.clientToolCallId(RAW),
    cursorToolCallId: RAW,
    toolName: "Bash",
    decodedArgs: "{}",
  });
  // listened like every live run, so closing the bridge releases the
  // resumed response's first-frame wait
  __test.listen(run as never, { buf: Buffer.alloc(0), refusal: null, closedCode: null });
  __test.resumeRun(run as never, [{ toolCallId: toolCallIdSentBack, content: "MARKER-42\n" }], "", true);
  fb.close();
  return mcpResultsOf(fb.writes);
}

describe("tool-call ids", () => {
  test("the id handed to the client is already in Claude's alphabet", () => {
    const id = __test.clientToolCallId(RAW);
    expect(id).toMatch(/^[A-Za-z0-9_-]+$/);
    expect(id).toBe(PROXY_REWRITE); // the proxy's rewrite is then a no-op
    expect(__test.clientToolCallId(id)).toBe(id);
  });

  test("a result sent back under the proxy-rewritten id reaches Cursor as success", () => {
    const [r] = resumeWith(PROXY_REWRITE);
    expect(r!.result.case).toBe("success");
    const text = (r!.result.value as { content: { content: { value: { text: string } } }[] }).content[0]!.content.value.text;
    expect(text).toBe("MARKER-42\n");
  });

  test("a result sent back under Cursor's raw id also matches", () => {
    const [r] = resumeWith(RAW);
    expect(r!.result.case).toBe("success");
  });

  test("a result for some other call is still an error, not a false match", () => {
    const [r] = resumeWith("call-other");
    expect(r!.result.case).toBe("error");
  });

  test("an id with no usable characters gets a fresh one", () => {
    expect(__test.clientToolCallId("\n\n")).toMatch(/^call_[0-9a-f]{32}$/);
  });
});
