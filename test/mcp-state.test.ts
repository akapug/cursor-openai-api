// helm arm (2026-09-25): Cursor asks for MCP state (exec field 36) before it
// runs an MCP tool. The bridge must answer on the same exec with field 36 =
// success{servers:[{name, identifier, tools}]}; an unanswered query stalled
// every grok-4.7 tool call until the h2 child's timer killed the stream.
import { describe, expect, test } from "bun:test";
import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import { BinaryReader, BinaryWriter, WireType } from "@bufbuild/protobuf/wire";
import {
  AgentClientMessageSchema,
  ExecServerMessageSchema,
  McpToolDefinitionSchema,
} from "../src/proto/agent_pb";
import { __test } from "../src/proxy";

function fields(bytes: Uint8Array): Array<{ no: number; bytes?: Uint8Array; text?: string }> {
  const r = new BinaryReader(bytes);
  const out: Array<{ no: number; bytes?: Uint8Array; text?: string }> = [];
  while (r.pos < r.len) {
    const [no, wt] = r.tag();
    if (wt !== WireType.LengthDelimited) { r.skip(wt, no); out.push({ no }); continue; }
    const b = r.bytes();
    out.push({ no, bytes: b, text: new TextDecoder().decode(b) });
  }
  return out;
}

describe("mcp state exec", () => {
  test("field-36 exec is answered with our server and its tools", () => {
    const head = toBinary(ExecServerMessageSchema, create(ExecServerMessageSchema, { id: 7, execId: "exec-1" }));
    const args = new BinaryWriter().tag(1, WireType.LengthDelimited).string("opencode").finish();
    const tail = new BinaryWriter().tag(36, WireType.LengthDelimited).bytes(args).tag(55, WireType.Varint).uint32(0).finish();
    const execMsg = fromBinary(ExecServerMessageSchema, new Uint8Array([...head, ...tail]));
    expect(execMsg.message.case).toBeUndefined();

    const tool = create(McpToolDefinitionSchema, { name: "Bash", toolName: "Bash", providerIdentifier: "opencode" });
    const sent: Uint8Array[] = [];
    __test.handleExecMessage(execMsg, [tool], (f) => sent.push(f), () => {
      throw new Error("not an mcp call");
    });
    expect(sent.length).toBe(1);

    const frame = sent[0]!;
    const client = fromBinary(AgentClientMessageSchema, frame.subarray(5));
    expect(client.message.case).toBe("execClientMessage");
    const exec = client.message.value as { id: number; execId: string; $unknown?: { no: number; data: Uint8Array }[] };
    expect(exec.id).toBe(7);
    expect(exec.execId).toBe("exec-1");
    const f36 = exec.$unknown?.find((u) => u.no === 36);
    expect(f36).toBeDefined();

    const result = new BinaryReader(f36!.data).bytes();
    const [success] = fields(result);
    expect(success!.no).toBe(1);
    const [servers] = fields(success!.bytes!);
    expect(servers!.no).toBe(1);
    const server = fields(servers!.bytes!);
    expect(server.filter((f) => f.no === 1).map((f) => f.text)).toEqual(["opencode"]);
    expect(server.filter((f) => f.no === 2).map((f) => f.text)).toEqual(["opencode"]);
    const tools = server.filter((f) => f.no === 5).map((f) => fromBinary(McpToolDefinitionSchema, f.bytes!));
    expect(tools.map((t) => t.name)).toEqual(["Bash"]);
  });

  test("an exec with an unknown field that is NOT 36 is still left alone", () => {
    const head = toBinary(ExecServerMessageSchema, create(ExecServerMessageSchema, { id: 8, execId: "exec-2" }));
    const tail = new BinaryWriter().tag(99, WireType.LengthDelimited).string("x").finish();
    const execMsg = fromBinary(ExecServerMessageSchema, new Uint8Array([...head, ...tail]));
    const sent: Uint8Array[] = [];
    __test.handleExecMessage(execMsg, [], (f) => sent.push(f), () => {});
    expect(sent.length).toBe(0);
  });
});
