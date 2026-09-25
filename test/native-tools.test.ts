// helm arm (2026-09-25): Cursor's own tools are refused so the model falls
// back to the caller's MCP tools, and each refusal must go out on the channel
// that exec reads: a streamed shell (shell_stream_args) is answered with a
// ShellStream event. Answering it with shell_result left Cursor waiting on
// heartbeats until the HTTP idle timeout (measured: 254 s, "unexpected EOF").
import { describe, expect, test } from "bun:test";
import { create, fromBinary } from "@bufbuild/protobuf";
import { AgentClientMessageSchema, ExecServerMessageSchema, ShellArgsSchema } from "../src/proto/agent_pb";
import { __test } from "../src/proxy";

function answer(kind: "shellArgs" | "shellStreamArgs") {
  const exec = create(ExecServerMessageSchema, {
    id: 4,
    execId: "exec-4",
    message: { case: kind, value: create(ShellArgsSchema, { command: "echo hi", workingDirectory: "/tmp" }) },
  });
  const sent: Uint8Array[] = [];
  __test.handleExecMessage(exec, [], (f) => sent.push(f), () => { throw new Error("not mcp"); });
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
});
