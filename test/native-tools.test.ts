// helm arm (2026-09-25): Cursor's own tools are refused so the model falls
// back to the caller's MCP tools, and each refusal must go out on the channel
// that exec reads: a streamed shell (shell_stream_args) is answered with a
// ShellStream event. Answering it with shell_result left Cursor waiting on
// heartbeats until the HTTP idle timeout (measured: 254 s, "unexpected EOF").
import { describe, expect, test } from "bun:test";
import { create, fromBinary } from "@bufbuild/protobuf";
import {
  AgentClientMessageSchema,
  ExecServerMessageSchema,
  GrepArgsSchema,
  McpToolDefinitionSchema,
  ReadArgsSchema,
  ShellArgsSchema,
  type ExecServerMessage,
  type McpToolDefinition,
} from "../src/proto/agent_pb";
import { __test } from "../src/proxy";

/** The caller's tools as the bridge registers them: server "opencode". */
function tools(...names: string[]): McpToolDefinition[] {
  return names.map((name) => create(McpToolDefinitionSchema, { name, toolName: name, providerIdentifier: "opencode" }));
}

function answer(kind: "shellArgs" | "shellStreamArgs", mcp: McpToolDefinition[] = []) {
  const exec = create(ExecServerMessageSchema, {
    id: 4,
    execId: "exec-4",
    message: { case: kind, value: create(ShellArgsSchema, { command: "echo hi", workingDirectory: "/tmp" }) },
  });
  return reply(exec, mcp);
}

function reply(exec: ExecServerMessage, mcp: McpToolDefinition[]) {
  const sent: Uint8Array[] = [];
  __test.handleExecMessage(exec, mcp, (f) => sent.push(f), () => { throw new Error("not mcp"); });
  expect(sent.length).toBe(1);
  const client = fromBinary(AgentClientMessageSchema, sent[0]!.subarray(5));
  return client.message.value as { id: number; execId: string; message: { case: string; value: any } };
}

describe("native tool refusals", () => {
  test("a streamed shell is refused with a ShellStream rejected event", () => {
    const a = answer("shellStreamArgs");
    expect([a.id, a.execId]).toEqual([4, "exec-4"]);
    expect(a.message.case).toBe("shellStream");
    expect(a.message.value.event.case).toBe("rejected");
    expect(a.message.value.event.value.command).toBe("echo hi");
  });

  test("a plain shell is still refused with a ShellResult", () => {
    const a = answer("shellArgs");
    expect(a.message.case).toBe("shellResult");
    expect(a.message.value.result.case).toBe("rejected");
  });

  // helm arm (2026-09-26): A REFUSAL NAMES THE ROUTE. MEASURED on the cursor
  // seat: grok-4.7-high read "Tool not available in this environment. Use the
  // MCP tools provided instead." as the end of the road and ended turns
  // saying every command channel was unavailable. Each refusal names the
  // caller's tool that does the same job, its server and the Cursor tool that
  // reaches it.
  test("a refused shell names the Bash MCP tool, its server and CallDynamicTool", () => {
    for (const kind of ["shellArgs", "shellStreamArgs"] as const) {
      const a = answer(kind, tools("Bash", "Read", "Grep"));
      const reason = kind === "shellArgs" ? a.message.value.result.value.reason : a.message.value.event.value.reason;
      expect(reason).toContain("CallDynamicTool");
      expect(reason).toContain("opencode");
      expect(reason).toContain('"Bash"');
      expect(reason).toContain("GetDynamicTools");
    }
  });

  test("a refused read or grep names the caller's own Read or Grep", () => {
    const mcp = tools("Bash", "Read", "Grep");
    const read = reply(create(ExecServerMessageSchema, {
      id: 5, execId: "exec-5", message: { case: "readArgs", value: create(ReadArgsSchema, { path: "/tmp/x" }) },
    }), mcp);
    expect(read.message.value.result.value.reason).toContain('"Read"');
    const grep = reply(create(ExecServerMessageSchema, {
      id: 6, execId: "exec-6", message: { case: "grepArgs", value: create(GrepArgsSchema, { pattern: "x" }) },
    }), mcp);
    expect(grep.message.value.result.value.error).toContain('"Grep"');
  });

  test("with no equivalent tool the refusal still names the server and the route", () => {
    const a = answer("shellArgs", tools("Monitor"));
    const reason: string = a.message.value.result.value.reason;
    expect(reason).toContain("CallDynamicTool");
    expect(reason).toContain("opencode");
    expect(reason).not.toContain('"Bash"');
  });
});
