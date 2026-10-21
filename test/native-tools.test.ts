// Regression test (2026-09-25): Cursor's own tools are refused so the model falls
// back to the caller's MCP tools, and each refusal must go out on the channel
// that exec reads: a streamed shell (shell_stream_args) is answered with a
// ShellStream event. Answering it with shell_result left Cursor waiting on
// heartbeats until the HTTP idle timeout (measured: 254 s, "unexpected EOF").
//
// Regression test (2026-09-28): A CURSOR TOOL THE CALLER CAN RUN IS RUN, NEVER
// REFUSED. MEASURED with CURSOR_BRIDGE_TRACE=1: grok-4.7-high opens a turn
// with Cursor's own read. The bridge refused it with prose that named the
// caller's Read. In a short conversation the model followed the prose; in a
// real Claude Code session's ~100k-token conversation it wrote text and ended
// the turn with no tool call. Once "the tool runtime refused every call" was
// in its history, every later turn gave up the same way for 30 hours. A
// native exec whose caller tool is registered now reaches the client as that
// tool's call, and the client's output reaches Cursor as the native's own
// result. Natives with no honest caller equivalent keep the refusal.
//
// Regression test (2026-09-29): A NATIVE READ BRINGS A BOUNDED NUMBER OF LINES INTO
// THE CLIENT'S CONTEXT. MEASURED on a live session's transcript: the model
// said it would read "in small slices", and each of Cursor's reads reached
// Claude Code as a Read with only file_path, so Claude Code read the whole
// file (a 100 KB module twice in 2 s, then 53 KB and 99 KB more). The
// conversation went from about 90k to 155k tokens in five seconds and
// compacted, every three minutes. Cursor's CLI proto carries the read's own
// offset and limit on ReadArgs fields 4 and 5, which the vendored proto
// predates: they go out as Read's offset and limit, never more than the
// bound, and a read the bound cut ends with the route to the rest.
//
// Review of that test: the line bound alone still let a 339-line, 99 KB page
// or one huge line through whole, and a negative offset came back as the
// file's first lines marked range_applied, the wrong slice as a success. The
// client's Read output is what fills its context, and the bridge cannot
// shrink what the client already read, so the bridge measures the file on
// disk and asks for no more lines than fit its byte bound; the text it
// returns to Cursor is held to that bound as well; a line that alone passes
// the bound, and a read from the end, are each answered with a read error
// naming the Shell route, with no client call.
//
// Second review: a read the bridge could not measure (a relative path, a
// file not on its disk, a window past its scan bound) still went out for 400
// lines, so a 100 KB line there reached the client whole. Such a read now
// FAILS CLOSED: its error names the route (the absolute path, or sed -n
// through the Shell tool), and the client is never called.
import { afterAll, afterEach, describe, expect, spyOn, test } from "bun:test";
import { create, fromBinary, fromJson, toBinary } from "@bufbuild/protobuf";
import { BinaryWriter, WireType } from "@bufbuild/protobuf/wire";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import {
  AgentClientMessageSchema,
  AgentServerMessageSchema,
  BackgroundShellSpawnArgsSchema,
  DeleteArgsSchema,
  ExecServerMessageSchema,
  FetchArgsSchema,
  GrepArgsSchema,
  LsArgsSchema,
  McpToolDefinitionSchema,
  ReadArgsSchema,
  ShellArgsSchema,
  WriteArgsSchema,
  type AgentServerMessage,
  type ExecServerMessage,
  type McpToolDefinition,
} from "../src/proto/agent_pb";
import { __test } from "../src/proxy";
import { fakeBridge, frame, makeRun, mcpCall, readSse } from "./helpers";

/** The caller's tools, with the input schemas Claude Code registers for them. */
const SCHEMAS: Record<string, Record<string, unknown>> = {
  Bash: {
    type: "object",
    properties: { command: { type: "string" }, timeout: { type: "number" }, description: { type: "string" } },
    required: ["command"],
  },
  Read: {
    type: "object",
    properties: { file_path: { type: "string" }, offset: { type: "number" }, limit: { type: "number" } },
    required: ["file_path"],
  },
  Grep: {
    type: "object",
    properties: Object.fromEntries(
      ["pattern", "path", "glob", "output_mode", "-B", "-A", "-C", "-n", "-i", "type", "head_limit", "multiline"]
        .map((k) => [k, {}]),
    ),
    required: ["pattern"],
  },
  Write: {
    type: "object",
    properties: { file_path: { type: "string" }, content: { type: "string" } },
    required: ["file_path", "content"],
  },
  WebFetch: {
    type: "object",
    properties: { url: { type: "string" }, prompt: { type: "string" } },
    required: ["url", "prompt"],
  },
  Monitor: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
};

/** The caller's tools as the bridge registers them: server "opencode", schema as sent. */
function tools(...names: string[]): McpToolDefinition[] {
  return names.map((name) => withSchema(name, SCHEMAS[name] ?? { type: "object", properties: {} }));
}

function withSchema(name: string, schema: Record<string, unknown>): McpToolDefinition {
  return create(McpToolDefinitionSchema, {
    name,
    toolName: name,
    providerIdentifier: "opencode",
    inputSchema: toBinary(ValueSchema, fromJson(ValueSchema, schema as never)),
  });
}

type Kind = NonNullable<ExecServerMessage["message"]["case"]>;

function exec(id: number, kind: Kind, value: unknown): ExecServerMessage {
  return create(ExecServerMessageSchema, { id, execId: `exec-${id}`, message: { case: kind, value } as never });
}

function server(e: ExecServerMessage): AgentServerMessage {
  return create(AgentServerMessageSchema, { message: { case: "execServerMessage", value: e } });
}

function answer(kind: "shellArgs" | "shellStreamArgs", mcp: McpToolDefinition[] = []) {
  return reply(exec(4, kind, create(ShellArgsSchema, { command: "echo hi", workingDirectory: "/tmp" })), mcp);
}

/** A refused exec: exactly one answer frame, and no tool call for the client. */
function reply(e: ExecServerMessage, mcp: McpToolDefinition[]) {
  const sent: Uint8Array[] = [];
  __test.handleExecMessage(e, mcp, (f) => sent.push(f), () => { throw new Error("not mcp"); });
  expect(sent.length).toBe(1);
  return clientMessage(sent[0]!).value as { id: number; execId: string; message: { case: string; value: any } };
}

function clientMessage(bytes: Uint8Array): { case: string; value: any } {
  return fromBinary(AgentClientMessageSchema, bytes.subarray(5)).message as { case: string; value: any };
}

type ToolCall = { toolName: string; toolCallId: string; decodedArgs: string };

/** What the handler does with an exec: the frames it sends Cursor and the tool calls it hands the client. */
function handled(e: ExecServerMessage, mcp: McpToolDefinition[]) {
  const sent: Uint8Array[] = [];
  const calls: ToolCall[] = [];
  __test.handleExecMessage(e, mcp, (f) => sent.push(f), (c) => calls.push(c as ToolCall));
  return { sent, calls, args: calls.map((c) => JSON.parse(c.decodedArgs) as Record<string, unknown>) };
}

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
const live: Array<ReturnType<typeof fakeBridge>> = [];
const savedLimits = { ...__test.limits };
afterEach(() => {
  for (const fb of live.splice(0)) fb.close(0);
  __test.pausedRuns.clear();
  Object.assign(__test.limits, savedLimits);
});
afterAll(() => rmSync(DIR, { recursive: true, force: true }));

/** A live run over a fake bridge whose request registered `mcp`. */
function started(mcp: McpToolDefinition[]) {
  const fb = fakeBridge();
  live.push(fb);
  const run = makeRun(fb.bridge) as { mcpTools: McpToolDefinition[] };
  run.mcpTools = mcp;
  const resp = __test.respond(run as never, true, () => __test.listen(run as never, NO_PEEK)) as Response;
  return { fb, run, resp };
}

/**
 * One exec (or several in one chunk) through a live run: the client's tool
 * calls, then the client's outputs sent back on the resumed request, and every
 * frame Cursor was sent for them.
 */
async function roundTrip(mcp: McpToolDefinition[], execs: ExecServerMessage[], outputs: Array<string | undefined>, userText = "") {
  const a = started(mcp);
  a.fb.feed(...execs.map((e) => frame(server(e))));
  expect(a.fb.writes.length).toBe(0);   // no refusal went to Cursor
  const s = await readSse(a.resp);
  expect(s.finish).toBe("tool_calls");
  const results = s.toolCalls.flatMap((c, i) => outputs[i] === undefined ? [] : [{ toolCallId: c.id, content: outputs[i]! }]);
  const pending = __test.resumeRun(a.run as never, results, userText, true);
  const sent = a.fb.writes.map(clientMessage);
  a.fb.close(0);
  if (pending) await pending;
  return { calls: s.toolCalls, sent };
}

/** The result a translated exec was answered with, by its ExecClientMessage case. */
function resultOf(sent: Array<{ case: string; value: any }>, execId: string) {
  const m = sent.find((x) => x.case === "execClientMessage" && x.value.execId === execId);
  expect(m).toBeDefined();
  return m!.value.message as { case: string; value: any };
}

/** READ_BOUND in src/proxy.ts: the most lines one native read brings into the client's context. */
const BOUND = 400;
/** READ_BYTE_BOUND in src/proxy.ts: the most bytes of a file's text one native read brings in. */
const BYTES = 24000;

/** Real files for the reads the bridge measures on disk, and a path that is not there. */
const DIR = mkdtempSync(join(tmpdir(), "bridge-read-bound-"));
const ABSENT = join(DIR, "absent.py");
/** A page of long lines: 339 lines of 291 characters, 99 KB, like a real Markdown page. */
const PAGE = join(DIR, "page.md");
writeFileSync(PAGE, Array.from({ length: 339 }, () => "p".repeat(291) + "\n").join(""));
/** One line of 100,000 bytes, then a short one. */
const ONE_LINE = join(DIR, "one-line.json");
writeFileSync(ONE_LINE, "x".repeat(100_000) + "\nshort\n");
const SHORT = join(DIR, "short.py");
writeFileSync(SHORT, Array.from({ length: 10 }, (_, i) => `line ${i + 1}\n`).join(""));
const NOTES = join(DIR, "notes.txt");
writeFileSync(NOTES, "hello\n");
const R = join(DIR, "r.txt");
writeFileSync(R, "r\n");
const BIG = join(DIR, "big.py");
writeFileSync(BIG, Array.from({ length: 2000 }, (_, i) => `line ${i + 1}\n`).join(""));
const LONG = join(DIR, "long.py");
writeFileSync(LONG, Array.from({ length: 1000 }, (_, i) => `line ${i + 1}\n`).join(""));

/** Claude Code's Read output for `path`'s lines `from`..`to`, read off the disk. */
function printed(path: string, from: number, to: number): string {
  const lines = readFileSync(path, "utf8").split("\n");
  return Array.from({ length: to - from + 1 }, (_, i) => `${from + i}\t${lines[from + i - 1]}`).join("\n");
}

const utf8 = (s: string): number => Buffer.byteLength(s, "utf8");

/**
 * A read as Cursor's server sends it: its offset (field 4) and limit
 * (field 5) are fields of Cursor's CLI proto that the vendored proto
 * predates, so the bridge decodes them as unknown fields.
 */
function readExec(id: number, path: string, range: { offset?: number; limit?: number } = {}): ExecServerMessage {
  const extra = new BinaryWriter();
  if (range.offset !== undefined) extra.tag(4, WireType.Varint).int32(range.offset);
  if (range.limit !== undefined) extra.tag(5, WireType.Varint).uint32(range.limit);
  const known = toBinary(ReadArgsSchema, create(ReadArgsSchema, { path, toolCallId: `call-${id}` }));
  return exec(id, "readArgs", fromBinary(ReadArgsSchema, new Uint8Array([...known, ...extra.finish()])));
}

/** Claude Code's Read output for lines `from`..`to`: "N<tab>text" per line. */
function numbered(from: number, to: number): string {
  return Array.from({ length: to - from + 1 }, (_, i) => `${from + i}\tline ${from + i}`).join("\n");
}

/** ReadSuccess.range_applied (field 8, unknown to the vendored proto): the bytes it carries, or undefined. */
function rangeApplied(success: { $unknown?: Array<{ no: number; data: Uint8Array }> }): number[] | undefined {
  const f = success.$unknown?.find((u) => u.no === 8);
  return f && [...f.data];
}

const ANY_ROUTE = `call a tool on MCP server "opencode" with CallDynamicTool (namespace "opencode"); GetDynamicTools lists them with their schemas`;
const BASH_ROUTE = `call the MCP tool "Bash" on server "opencode" with CallDynamicTool (namespace "opencode", toolName "Bash"); GetDynamicTools gives its schema`;
/** A refusal as the model reads it: Cursor's tool, the one reason, and the route. */
const refusal = (tool: string, why: string, route: string): string =>
  `Cursor's ${tool} is not run in this session: ${why}. Do not stop here: ${route}.`;
const BACKGROUND = "a background shell is one Cursor addresses later by a shell id the caller's Bash never gives; run it with Bash and run_in_background";

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

  // Regression test (2026-09-26): A REFUSAL NAMES THE ROUTE. MEASURED on a live
  // session: grok-4.7-high read "Tool not available in this environment. Use the
  // MCP tools provided instead." as the end of the road and ended turns
  // saying every command channel was unavailable. Each refusal names the
  // caller's tool that does the same job, its server and the Cursor tool that
  // reaches it. (2026-09-28: a shell with Bash registered is run, not
  // refused, so the refusals that still name Bash are the ones kept.)
  test("a refused background shell or delete names the Bash MCP tool, its server and CallDynamicTool", () => {
    const mcp = tools("Bash", "Read", "Grep");
    const background = reply(exec(5, "backgroundShellSpawnArgs",
      create(BackgroundShellSpawnArgsSchema, { command: "sleep 9", workingDirectory: "/tmp" })), mcp);
    const del = reply(exec(6, "deleteArgs", create(DeleteArgsSchema, { path: "/tmp/x" })), mcp);
    for (const reason of [background.message.value.result.value.reason, del.message.value.result.value.reason]) {
      expect(reason).toContain("CallDynamicTool");
      expect(reason).toContain("opencode");
      expect(reason).toContain('"Bash"');
      expect(reason).toContain("GetDynamicTools");
    }
  });

  test("a Read or Grep whose schema cannot take the call is refused, and the refusal names it", () => {
    // opencode's read takes filePath and its grep has no output_mode: the
    // bridge cannot fill them, so it names the tool instead of calling it.
    const mcp = [
      withSchema("Read", { type: "object", properties: { filePath: { type: "string" } }, required: ["filePath"] }),
      withSchema("Grep", { type: "object", properties: { pattern: {}, include: {} }, required: ["pattern"] }),
    ];
    const read = reply(exec(5, "readArgs", create(ReadArgsSchema, { path: NOTES })), mcp);
    expect(read.message.value.result.case).toBe("rejected");
    expect(read.message.value.result.value.reason).toContain('"Read"');
    const grep = reply(exec(6, "grepArgs", create(GrepArgsSchema, { pattern: "x" })), mcp);
    expect(grep.message.value.result.value.error).toContain('"Grep"');
  });

  test("with no equivalent tool the refusal still names the server and the route", () => {
    const a = answer("shellArgs", tools("Monitor"));
    const reason: string = a.message.value.result.value.reason;
    expect(reason).toContain("CallDynamicTool");
    expect(reason).toContain("opencode");
    expect(reason).not.toContain('"Bash"');
  });

  test("a native with no registered equivalent is refused exactly as before", () => {
    const mcp = tools("Monitor");
    const grep = reply(exec(7, "grepArgs", create(GrepArgsSchema, { pattern: "TODO" })), mcp);
    expect(grep.message.case).toBe("grepResult");
    expect(grep.message.value.result.case).toBe("error");
    expect(grep.message.value.result.value.error).toBe(refusal("Grep", "this session registers no Grep or Bash tool", ANY_ROUTE));
    const shell = answer("shellArgs", mcp).message.value.result;
    expect(shell.case).toBe("rejected");
    expect([shell.value.command, shell.value.workingDirectory, shell.value.reason, shell.value.isReadonly])
      .toEqual(["echo hi", "/tmp", refusal("Shell", "this session registers no Bash tool", ANY_ROUTE), false]);
    const read = reply(exec(8, "readArgs", create(ReadArgsSchema, { path: "/tmp/x" })), mcp);
    expect(read.message.value.result.value).toMatchObject({ path: "/tmp/x", reason: refusal("Read", "this session registers no Read tool", ANY_ROUTE) });
  });

  test("a delete stays refused with the existing reason, even with Bash registered", () => {
    const del = reply(exec(9, "deleteArgs", create(DeleteArgsSchema, { path: "/tmp/x" })), tools("Bash", "Read", "Write"));
    expect(del.message.case).toBe("deleteResult");
    expect(del.message.value.result.case).toBe("rejected");
    expect(del.message.value.result.value).toMatchObject({ path: "/tmp/x", reason: refusal("Delete", "a delete has no caller tool that runs it", BASH_ROUTE) });
  });

  test("a read with Bash but no Read registered stays refused and names Bash; it is never run as cat", () => {
    const { sent, calls } = handled(exec(10, "readArgs", create(ReadArgsSchema, { path: "/tmp/x" })), tools("Bash"));
    expect(calls).toEqual([]);
    expect(sent.length).toBe(1);
    const read = clientMessage(sent[0]!).value.message;
    expect(read.case).toBe("readResult");
    expect(read.value.result.case).toBe("rejected");
    expect(read.value.result.value.reason).toBe(refusal("Read", "this session registers no Read tool", BASH_ROUTE));
  });

  test("a fetch with Bash but no WebFetch registered stays refused", () => {
    const { sent, calls } = handled(exec(11, "fetchArgs", create(FetchArgsSchema, { url: "https://example.com" })), tools("Bash"));
    expect(calls).toEqual([]);
    expect(clientMessage(sent[0]!).value.message.value.result.case).toBe("error");
  });

  test("a background shell stays refused: Cursor addresses it later by a shell id the caller's Bash never gives", () => {
    const { sent, calls } = handled(exec(13, "shellArgs",
      create(ShellArgsSchema, { command: "npm run dev", isBackground: true })), tools("Bash"));
    expect(calls).toEqual([]);
    expect(clientMessage(sent[0]!).value.message.value.result.value.reason).toBe(refusal("Shell", BACKGROUND, BASH_ROUTE));
  });

  test("a write of raw bytes stays refused: the caller's Write takes text", () => {
    const { calls } = handled(exec(12, "writeArgs",
      create(WriteArgsSchema, { path: "/tmp/b.bin", fileBytes: new Uint8Array([0, 1, 2]) })), tools("Write", "Bash"));
    expect(calls).toEqual([]);
  });
});

describe("native tools the caller can run are run as the caller's tool", () => {
  test("a read with Read registered reaches the client as a Read call, and Cursor is sent no refusal", async () => {
    const a = started(tools("Bash", "Read", "Grep"));
    a.fb.feed(frame(server(exec(9, "readArgs", create(ReadArgsSchema, { path: NOTES, toolCallId: "call-r1\nfc_r1" })))));
    expect(a.fb.writes.length).toBe(0);
    const s = await readSse(a.resp);
    expect(s.finish).toBe("tool_calls");
    expect(s.toolCalls.map((c) => [c.id, c.name, JSON.parse(c.arguments)]))
      .toEqual([["call-r1_fc_r1", "Read", { file_path: NOTES, limit: BOUND }]]);
    expect(a.fb.writes.length).toBe(0);
    expect(__test.pausedRuns.get("call-r1_fc_r1")).toBe(a.run as never);
  });

  test("the client's Read output reaches Cursor as the read's own success, carrying the text", async () => {
    const { sent } = await roundTrip(tools("Read"),
      [exec(9, "readArgs", create(ReadArgsSchema, { path: NOTES, toolCallId: "call-r1" }))], ["     1\thello\n"]);
    const cursor = sent.find((x) => x.case === "execClientMessage")!;
    expect([cursor.value.id, cursor.value.execId]).toEqual([9, "exec-9"]);
    const read = resultOf(sent, "exec-9");
    expect(read.case).toBe("readResult");
    expect(read.value.result.case).toBe("success");
    expect(read.value.result.value.path).toBe(NOTES);
    expect(read.value.result.value.output).toEqual({ case: "content", value: "     1\thello\n" });
    expect(read.value.result.value.totalLines).toBe(1);
    expect(sent.some((x) => x.value?.message?.case === "mcpResult")).toBe(false);
  });

  test("a Read the client refused reaches Cursor as the read's error, carrying why", async () => {
    const { sent } = await roundTrip(tools("Read"), [exec(9, "readArgs", create(ReadArgsSchema, { path: NOTES }))],
      ["<tool_use_error>Read of this file was denied by a PreToolUse hook.</tool_use_error>"]);
    const read = resultOf(sent, "exec-9").value.result;
    expect(read.case).toBe("error");
    expect(read.value).toMatchObject({ path: NOTES, error: "Read of this file was denied by a PreToolUse hook." });
  });

  test("a read whose result never came back is answered as the read's error, not left open", async () => {
    const { sent } = await roundTrip(tools("Read"), [exec(9, "readArgs", create(ReadArgsSchema, { path: NOTES }))], [undefined]);
    const read = resultOf(sent, "exec-9").value.result;
    expect(read.case).toBe("error");
    expect(read.value.error).toBe("Tool result not provided");
  });

  test("a shell with Bash registered runs as Bash in its working directory, and its output is the shell's success", async () => {
    const { calls, sent } = await roundTrip(tools("Bash", "Read"),
      [exec(4, "shellArgs", create(ShellArgsSchema, { command: "echo hi", workingDirectory: "/tmp/a b" }))], ["hi\n"]);
    expect(calls.map((c) => [c.name, JSON.parse(c.arguments)])).toEqual([["Bash", { command: "cd '/tmp/a b' && echo hi" }]]);
    const shell = resultOf(sent, "exec-4");
    expect(shell.case).toBe("shellResult");
    expect(shell.value.result.case).toBe("success");
    expect(shell.value.result.value).toMatchObject({ command: "echo hi", workingDirectory: "/tmp/a b", exitCode: 0, stdout: "hi\n" });
  });

  test("a shell the client reports as a non-zero exit is the shell's failure with that exit code", async () => {
    const { sent } = await roundTrip(tools("Bash"),
      [exec(4, "shellArgs", create(ShellArgsSchema, { command: "false" }))], ["Exit code 2\nboom\n"]);
    const shell = resultOf(sent, "exec-4").value.result;
    expect(shell.case).toBe("failure");
    expect(shell.value).toMatchObject({ command: "false", exitCode: 2, stdout: "boom\n" });
  });

  test("a working directory with a quote in it is quoted for the shell, and none leaves the command alone", () => {
    const quoted = handled(exec(4, "shellArgs", create(ShellArgsSchema, { command: "ls", workingDirectory: "/tmp/it's" })), tools("Bash"));
    expect(quoted.args).toEqual([{ command: "cd '/tmp/it'\\''s' && ls" }]);
    expect(quoted.sent.length).toBe(0);
    const bare = handled(exec(5, "shellArgs", create(ShellArgsSchema, { command: "ls" })), tools("Bash"));
    expect(bare.args).toEqual([{ command: "ls" }]);
  });

  test("a streamed shell runs as Bash and is answered on its stream: start, output, exit, then close", async () => {
    const { calls, sent } = await roundTrip(tools("Bash"),
      [exec(6, "shellStreamArgs", create(ShellArgsSchema, { command: "echo hi", workingDirectory: "/tmp" }))], ["hi\n"]);
    expect(calls.map((c) => [c.name, JSON.parse(c.arguments)])).toEqual([["Bash", { command: "cd '/tmp' && echo hi" }]]);
    const events = sent.filter((x) => x.case === "execClientMessage").map((x) => {
      expect([x.value.id, x.value.execId, x.value.message.case]).toEqual([6, "exec-6", "shellStream"]);
      return x.value.message.value.event as { case: string; value: any };
    });
    expect(events.map((e) => e.case)).toEqual(["start", "stdout", "exit"]);
    expect(events[1]!.value.data).toBe("hi\n");
    expect(events[2]!.value).toMatchObject({ code: 0, cwd: "/tmp", aborted: false });
    const last = sent.at(-1)!;
    expect(last.case).toBe("execClientControlMessage");
    expect(last.value.message).toMatchObject({ case: "streamClose", value: { id: 6 } });
  });

  test("a grep runs as the caller's Grep with the fields its schema takes, content mode by default", () => {
    const e = exec(7, "grepArgs", create(GrepArgsSchema, {
      pattern: "TODO", path: "src", glob: "*.ts", caseInsensitive: true, contextAfter: 1, headLimit: 20, sort: "none",
    }));
    const full = handled(e, tools("Grep"));
    expect(full.sent.length).toBe(0);
    expect(full.calls.map((c) => c.toolName)).toEqual(["Grep"]);
    expect(full.args).toEqual([{
      pattern: "TODO", output_mode: "content", path: "src", glob: "*.ts", "-A": 1, "-i": true, head_limit: 20, "-n": true,
    }]);
    const narrow = handled(e, [withSchema("Grep", {
      type: "object", properties: { pattern: {}, path: {}, output_mode: {} }, required: ["pattern"],
    })]);
    expect(narrow.args).toEqual([{ pattern: "TODO", output_mode: "content", path: "src" }]);
  });

  test("the client's Grep lines reach Cursor as the grep's success, split by file with line numbers", async () => {
    // A context line whose text holds ":12:" also reads as a match line of the
    // made-up path "src/a.ts-2-see b.ts"; it belongs to src/a.ts.
    const { sent } = await roundTrip(tools("Grep"), [exec(7, "grepArgs", create(GrepArgsSchema, { pattern: "TODO", path: "src", context: 1 }))],
      ["src/a.ts-2-see b.ts:12:x\nsrc/a.ts:3:// TODO one\nsrc/a.ts-4-next: line\n--\nsrc/b-2-c.ts:10:x TODO\n"]);
    const grep = resultOf(sent, "exec-7");
    expect(grep.case).toBe("grepResult");
    expect(grep.value.result.case).toBe("success");
    const ok = grep.value.result.value;
    expect([ok.pattern, ok.path, ok.outputMode]).toEqual(["TODO", "src", "content"]);
    const results = Object.values(ok.workspaceResults) as Array<{ result: { case: string; value: any } }>;
    expect(results.length).toBe(1);
    expect(results[0]!.result.case).toBe("content");
    const content = results[0]!.result.value;
    expect(content.matches.map((f: any) => [f.file, f.matches.map((m: any) => [m.lineNumber, m.content, m.isContextLine])])).toEqual([
      ["src/a.ts", [[2, "see b.ts:12:x", true], [3, "// TODO one", false], [4, "next: line", true]]],
      ["src/b-2-c.ts", [[10, "x TODO", false]]],
    ]);
    expect([content.totalLines, content.totalMatchedLines]).toEqual([4, 2]);
  });

  test("a grep of one file, and one that matched nothing, are the grep's success too", async () => {
    const one = await roundTrip(tools("Grep"), [exec(7, "grepArgs", create(GrepArgsSchema, { pattern: "x", path: "/tmp/f.ts" }))],
      ["12:let x = 1\n13-y\n"]);
    const file = Object.values(resultOf(one.sent, "exec-7").value.result.value.workspaceResults)[0] as any;
    expect(file.result.value.matches.map((f: any) => [f.file, f.matches.map((m: any) => [m.lineNumber, m.isContextLine])]))
      .toEqual([["/tmp/f.ts", [[12, false], [13, true]]]]);
    const none = await roundTrip(tools("Grep"), [exec(8, "grepArgs", create(GrepArgsSchema, { pattern: "zzz" }))], ["No matches found"]);
    const empty = resultOf(none.sent, "exec-8").value.result;
    expect(empty.case).toBe("success");
    expect((Object.values(empty.value.workspaceResults)[0] as any).result.value.matches).toEqual([]);
  });

  test("a files or count grep reaches Cursor as the files or counts the client listed", async () => {
    const files = await roundTrip(tools("Grep"),
      [exec(7, "grepArgs", create(GrepArgsSchema, { pattern: "x", outputMode: "files_with_matches" }))], ["Found 2 files\nsrc/a.ts\nsrc/b.ts"]);
    const f = (Object.values(resultOf(files.sent, "exec-7").value.result.value.workspaceResults)[0] as any).result;
    expect(f.case).toBe("files");
    expect([f.value.files, f.value.totalFiles]).toEqual([["src/a.ts", "src/b.ts"], 2]);
    const counts = await roundTrip(tools("Grep"),
      [exec(8, "grepArgs", create(GrepArgsSchema, { pattern: "x", outputMode: "count" }))],
      ["src/a.ts:2\nsrc/b.ts:1\n\nFound 3 total occurrences across 2 files."]);
    const c = (Object.values(resultOf(counts.sent, "exec-8").value.result.value.workspaceResults)[0] as any).result;
    expect(c.case).toBe("count");
    expect(c.value.counts.map((x: any) => [x.file, x.count])).toEqual([["src/a.ts", 2], ["src/b.ts", 1]]);
    expect([c.value.totalFiles, c.value.totalMatches]).toEqual([2, 3]);
  });

  test("Grep output the bridge cannot split is the grep's error carrying all of it, never a made-up match list", async () => {
    const text = "<persisted-output>\nOutput too large (40.8KB). Full output saved to: /tmp/out.txt\n</persisted-output>";
    const { sent } = await roundTrip(tools("Grep"), [exec(7, "grepArgs", create(GrepArgsSchema, { pattern: "x" }))], [text]);
    const grep = resultOf(sent, "exec-7").value.result;
    expect(grep.case).toBe("error");
    expect(grep.value.error).toContain(text);
  });

  test("a write with Write registered runs as Write, and a confirmed write is the write's success", async () => {
    const { calls, sent } = await roundTrip(tools("Write", "Bash"),
      [exec(5, "writeArgs", create(WriteArgsSchema, { path: "/tmp/w.txt", fileText: "a\nb\n" }))], ["File created successfully at: /tmp/w.txt"]);
    expect(calls.map((c) => [c.name, JSON.parse(c.arguments)])).toEqual([["Write", { file_path: "/tmp/w.txt", content: "a\nb\n" }]]);
    const write = resultOf(sent, "exec-5");
    expect(write.case).toBe("writeResult");
    expect(write.value.result.case).toBe("success");
    expect(write.value.result.value).toMatchObject({ path: "/tmp/w.txt", linesCreated: 2, fileSize: 4 });
  });

  test("a write the client did not confirm is the write's error carrying its words, never a success", async () => {
    const refused = await roundTrip(tools("Write"), [exec(5, "writeArgs", create(WriteArgsSchema, { path: "/tmp/w.txt", fileText: "a" }))],
      ["<tool_use_error>File has not been read yet. Read it first before writing to it.</tool_use_error>"]);
    expect(resultOf(refused.sent, "exec-5").value.result).toMatchObject({
      case: "error", value: { path: "/tmp/w.txt", error: "File has not been read yet. Read it first before writing to it." },
    });
    const unclear = await roundTrip(tools("Write"), [exec(6, "writeArgs", create(WriteArgsSchema, { path: "/tmp/w.txt", fileText: "a" }))],
      ["Execution stopped by PreToolUse hook: no writes here"]);
    expect(resultOf(unclear.sent, "exec-6").value.result).toMatchObject({
      case: "error", value: { error: "Execution stopped by PreToolUse hook: no writes here" },
    });
  });

  test("a directory listing runs as Bash ls, and its names reach Cursor as the listing's tree", async () => {
    const { calls, sent } = await roundTrip(tools("Bash"), [exec(3, "lsArgs", create(LsArgsSchema, { path: "/tmp/d" }))],
      ["src/\nREADME.md\n.env\n-- end of listing --\n"]);
    expect(calls.map((c) => [c.name, JSON.parse(c.arguments)]))
      .toEqual([["Bash", { command: "ls -1Ap -- '/tmp/d' && echo '-- end of listing --'" }]]);
    const ls = resultOf(sent, "exec-3");
    expect(ls.case).toBe("lsResult");
    expect(ls.value.result.case).toBe("success");
    const root = ls.value.result.value.directoryTreeRoot;
    expect(root.absPath).toBe("/tmp/d");
    expect(root.childrenDirs.map((d: any) => d.absPath)).toEqual(["/tmp/d/src"]);
    expect(root.childrenFiles.map((f: any) => f.name)).toEqual(["README.md", ".env"]);
    expect(root.numFiles).toBe(2);
  });

  test("a listing that failed or was cut short is the listing's error carrying the client's words", async () => {
    const failed = await roundTrip(tools("Bash"), [exec(3, "lsArgs", create(LsArgsSchema, { path: "/nope" }))],
      ["Exit code 2\nls: cannot access '/nope': No such file or directory"]);
    expect(resultOf(failed.sent, "exec-3").value.result).toMatchObject({
      case: "error", value: { path: "/nope", error: "Exit code 2\nls: cannot access '/nope': No such file or directory" },
    });
    const cut = await roundTrip(tools("Bash"), [exec(4, "lsArgs", create(LsArgsSchema, { path: "/big" }))],
      ["<persisted-output>\nOutput too large (90KB). Full output saved to: /tmp/o.txt\n</persisted-output>"]);
    expect(resultOf(cut.sent, "exec-4").value.result.case).toBe("error");
  });

  test("a fetch with WebFetch registered runs as WebFetch, and its answer is the fetch's success", async () => {
    const { calls, sent } = await roundTrip(tools("WebFetch", "Bash"),
      [exec(2, "fetchArgs", create(FetchArgsSchema, { url: "https://example.com/doc" }))], ["# Doc\nbody"]);
    expect(calls.map((c) => [c.name, JSON.parse(c.arguments)]))
      .toEqual([["WebFetch", { url: "https://example.com/doc", prompt: "Return the page content." }]]);
    const fetch = resultOf(sent, "exec-2");
    expect(fetch.case).toBe("fetchResult");
    expect(fetch.value.result).toMatchObject({ case: "success", value: { url: "https://example.com/doc", content: "# Doc\nbody" } });
  });

  test("text sent with the results rides a result that carries text, never a listing or a write", async () => {
    const { sent } = await roundTrip(tools("Read", "Write"), [
      exec(1, "readArgs", create(ReadArgsSchema, { path: R, toolCallId: "call-1" })),
      exec(2, "writeArgs", create(WriteArgsSchema, { path: "/tmp/w.txt", fileText: "w", toolCallId: "call-2" })),
    ], ["r", "The file /tmp/w.txt has been updated successfully."], "stop and summarize");
    expect(resultOf(sent, "exec-1").value.result.value.output.value).toBe("r\n\n<user_message>\nstop and summarize\n</user_message>");
    expect(resultOf(sent, "exec-2").value.result.case).toBe("success");
  });

  test("a translated native and an MCP call in one chunk each get their own kind of answer", async () => {
    const { calls, sent } = await roundTrip(tools("Read", "Bash"), [
      exec(1, "readArgs", create(ReadArgsSchema, { path: R, toolCallId: "call-1" })),
      mcpCall(2, "call-2", "echo 2").message.value as ExecServerMessage,
    ], ["r", "2\n"]);
    expect(calls.map((c) => c.name)).toEqual(["Read", "Bash"]);
    expect(resultOf(sent, "exec-1").case).toBe("readResult");
    expect(resultOf(sent, "exec-2").case).toBe("mcpResult");
  });
});

describe("a native read brings at most the bound's lines and bytes into the client's context", () => {
  test("a read with no range reaches the client as Read with the line bound as its limit", () => {
    const { sent, calls, args } = handled(readExec(9, BIG), tools("Read"));
    expect(sent.length).toBe(0);
    expect(calls.map((c) => c.toolName)).toEqual(["Read"]);
    expect(args).toEqual([{ file_path: BIG, limit: BOUND }]);
  });

  test("the read's own range reaches the client as Read's offset and limit, never over the bound", () => {
    const read = (range: { offset?: number; limit?: number }) => handled(readExec(9, BIG, range), tools("Read")).args;
    expect(read({ offset: 1200, limit: 80 })).toEqual([{ file_path: BIG, offset: 1200, limit: 80 }]);
    expect(read({ offset: 1200 })).toEqual([{ file_path: BIG, offset: 1200, limit: BOUND }]);
    expect(read({ offset: 1200, limit: 5000 })).toEqual([{ file_path: BIG, offset: 1200, limit: BOUND }]);
    expect(read({ limit: 30 })).toEqual([{ file_path: BIG, limit: 30 }]);
    // Line 1 (or Cursor's 0, which its CLI reads as line 1) needs no offset.
    expect(read({ offset: 1, limit: 30 })).toEqual([{ file_path: BIG, limit: 30 }]);
    expect(read({ offset: 0, limit: 30 })).toEqual([{ file_path: BIG, limit: 30 }]);
  });

  test("a Read whose schema takes no limit is refused and named, never called unbounded", () => {
    const mcp = [withSchema("Read", { type: "object", properties: { file_path: { type: "string" } }, required: ["file_path"] })];
    const { sent, calls } = handled(readExec(9, BIG), mcp);
    expect(calls).toEqual([]);
    const read = clientMessage(sent[0]!).value.message;
    expect(read.value.result.case).toBe("rejected");
    expect(read.value.result.value.reason).toContain('"Read"');
  });

  test("a small file reads whole exactly as today: its text, no note, not truncated, no range applied", async () => {
    const text = printed(SHORT, 1, 10);
    const { calls, sent } = await roundTrip(tools("Read"), [readExec(9, SHORT)], [text]);
    expect(calls.map((c) => JSON.parse(c.arguments))).toEqual([{ file_path: SHORT, limit: BOUND }]);
    const ok = resultOf(sent, "exec-9").value.result;
    expect(ok.case).toBe("success");
    expect(ok.value.output).toEqual({ case: "content", value: text });
    expect([ok.value.totalLines, ok.value.truncated]).toEqual([10, false]);
    expect(rangeApplied(ok.value)).toBeUndefined();
  });

  test("a read the line bound cut reaches Cursor with the line it stopped at and the Shell route to the rest", async () => {
    const text = printed(LONG, 1, BOUND);
    const { calls, sent } = await roundTrip(tools("Read"), [readExec(9, LONG)], [text]);
    expect(calls.map((c) => JSON.parse(c.arguments))).toEqual([{ file_path: LONG, limit: BOUND }]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    const content: string = ok.output.value;
    expect(content.startsWith(text)).toBe(true);
    const note = content.slice(text.length);
    expect(note).toContain(`stops at line ${BOUND}: the bridge brings at most ${BOUND} lines per read`);
    expect(note).toContain(`sed -n '${BOUND + 1},${2 * BOUND}p' '${LONG}'`);
    expect(note).toContain(`git grep -n PATTERN '${LONG}'`);
    expect(note).toContain(`wc -l '${LONG}'`);
    expect(note).toContain("Shell");
    expect(ok.truncated).toBe(true);
    // No range was asked for, so there is none Cursor could apply again.
    expect(rangeApplied(ok)).toBeUndefined();
  });

  // F1: 339 lines of 291 characters. Read prints "N<tab>text<newline>", so
  // lines 1-9 cost 294 bytes and lines 10-99 cost 295: 9 * 294 + 72 * 295 =
  // 23,886 fits 24,000, and an 82nd line (24,181) does not.
  test("a page of long lines is read only as far as the byte bound, so the client reads 81 of its 339 lines", async () => {
    const { calls, sent } = await roundTrip(tools("Read"), [readExec(9, PAGE)], [printed(PAGE, 1, 81)]);
    expect(calls.map((c) => JSON.parse(c.arguments))).toEqual([{ file_path: PAGE, limit: 81 }]);
    const content: string = resultOf(sent, "exec-9").value.result.value.output.value;
    const text = printed(PAGE, 1, 81);
    expect(utf8(text)).toBeLessThanOrEqual(BYTES);
    expect(content.startsWith(text)).toBe(true);
    const note = content.slice(text.length);
    expect(note).toContain(`stops at line 81: the next line would pass the ${BYTES} bytes the bridge brings per read`);
    expect(note).toContain(`sed -n '82,${81 + BOUND}p' '${PAGE}'`);
    expect(resultOf(sent, "exec-9").value.result.value.truncated).toBe(true);
  });

  test("a window the page ends inside keeps the line limit, and gets no note", async () => {
    const { calls, sent } = await roundTrip(tools("Read"), [readExec(9, PAGE, { offset: 320 })], [printed(PAGE, 320, 339)]);
    expect(calls.map((c) => JSON.parse(c.arguments))).toEqual([{ file_path: PAGE, offset: 320, limit: BOUND }]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    expect(ok.output.value).toBe(printed(PAGE, 320, 339));
    expect(ok.truncated).toBe(false);
    expect(rangeApplied(ok)).toEqual([1]);
  });

  test("a line that alone passes the byte bound is never read: Cursor gets a read error naming the Shell route", () => {
    const { sent, calls } = handled(readExec(9, ONE_LINE), tools("Read"));
    expect(calls).toEqual([]);
    expect(sent.length).toBe(1);
    const read = clientMessage(sent[0]!).value;
    expect([read.id, read.execId, read.message.case]).toEqual([9, "exec-9", "readResult"]);
    expect(read.message.value.result.case).toBe("error");
    const error: string = read.message.value.result.value.error;
    expect(error).toContain(`Line 1 of ${ONE_LINE} is longer than the ${BYTES} bytes`);
    expect(error).toContain(`sed -n '1p' '${ONE_LINE}' | cut -b 1-${BYTES}`);
    expect(error).toContain("Shell");
    // The line after it reads as usual.
    expect(handled(readExec(10, ONE_LINE, { offset: 2 }), tools("Read")).args)
      .toEqual([{ file_path: ONE_LINE, offset: 2, limit: BOUND }]);
  });

  // The file grew between the bridge's measure and the client's read: the
  // client's output is larger than the measure allowed.
  test("text that outgrew the measure is held to the byte bound on its way to Cursor, cut at a whole line", async () => {
    const line = (n: number) => `${n}\t${"w".repeat(300)}`;
    const text = Array.from({ length: 100 }, (_, i) => line(i + 1)).join("\n");
    const { sent } = await roundTrip(tools("Read"), [readExec(9, SHORT)], [text]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    const content: string = ok.output.value;
    const at = content.indexOf("\n\n[");
    const kept = content.slice(0, at);
    expect(utf8(kept)).toBeLessThanOrEqual(BYTES);
    const last = kept.split("\n").length;   // whole lines 1..last: each 303 to 305 bytes
    expect(kept).toBe(Array.from({ length: last }, (_, i) => line(i + 1)).join("\n"));
    expect(utf8(kept + "\n" + line(last + 1))).toBeGreaterThan(BYTES);
    expect(content.slice(at)).toContain(`stops at line ${last}: the text was cut to the ${BYTES} bytes the bridge brings per read`);
    expect(content.slice(at)).toContain(`sed -n '${last + 1},${last + BOUND}p' '${SHORT}'`);
    expect(ok.truncated).toBe(true);
  });

  test("one line that outgrew the measure reaches Cursor cut at a character, with the bytes it holds named", async () => {
    const text = `1\t${"é".repeat(30_000)}`;   // 60,000 bytes of text
    const { sent } = await roundTrip(tools("Read"), [readExec(9, SHORT)], [text]);
    const content: string = resultOf(sent, "exec-9").value.result.value.output.value;
    const at = content.indexOf("\n\n[");
    const kept = content.slice(0, at);
    expect(utf8(kept)).toBeLessThanOrEqual(BYTES);
    expect(kept).not.toContain("�");
    const held = utf8(kept) - utf8("1\t");
    expect(held).toBe(BYTES - 2);   // "1<tab>" plus 11,999 two-byte characters
    expect(content.slice(at)).toContain(`stops partway through line 1: it holds the first ${held} bytes of that line`);
    expect(content.slice(at)).toContain(`sed -n '1p' '${SHORT}' | cut -b ${held + 1}-${held + BYTES}`);
  });

  test("a read of its own range gets exactly those lines, marked applied so Cursor does not cut them again", async () => {
    const text = printed(LONG, 200, 279);
    const { calls, sent } = await roundTrip(tools("Read"), [readExec(9, LONG, { offset: 200, limit: 80 })], [text]);
    expect(calls.map((c) => JSON.parse(c.arguments))).toEqual([{ file_path: LONG, offset: 200, limit: 80 }]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    expect(ok.output.value).toBe(text);
    expect(ok.truncated).toBe(false);
    expect(rangeApplied(ok)).toEqual([1]);
  });

  test("a range longer than the line bound stops at the bound, with the next lines named", async () => {
    const text = printed(LONG, 200, 200 + BOUND - 1);
    const { sent } = await roundTrip(tools("Read"), [readExec(9, LONG, { offset: 200, limit: 700 })], [text]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    expect(ok.output.value).toContain(`stops at line ${200 + BOUND - 1}: the bridge brings at most ${BOUND} lines per read`);
    expect(ok.output.value).toContain(`sed -n '${200 + BOUND},${200 + 2 * BOUND - 1}p' '${LONG}'`);
    expect(ok.truncated).toBe(true);
    expect(rangeApplied(ok)).toEqual([1]);
  });

  test("a range the file ends inside gets no note", async () => {
    const text = printed(LONG, 990, 1000);
    const { sent } = await roundTrip(tools("Read"), [readExec(9, LONG, { offset: 990, limit: 1000 })], [text]);
    const ok = resultOf(sent, "exec-9").value.result.value;
    expect(ok.output.value).toBe(text);
    expect(ok.truncated).toBe(false);
  });

  // A read the bridge cannot measure fails closed: the fit to the byte bound
  // is not proven, so the client is never called.
  test("a relative path is answered with a read error naming the absolute path and the Shell route, with no client call", () => {
    const { sent, calls } = handled(readExec(9, "notes.txt"), tools("Read"));
    expect(calls).toEqual([]);
    expect(sent.length).toBe(1);
    const result = clientMessage(sent[0]!).value.message.value.result;
    expect(result.case).toBe("error");
    expect(result.value.path).toBe("notes.txt");
    expect(result.value.error).toContain("not an absolute path on this machine");
    expect(result.value.error).toContain("by its absolute path");
    expect(result.value.error).toContain(`sed -n '1,${BOUND}p' 'notes.txt' | cut -b 1-${BYTES}`);
  });

  test("a file not on the bridge's disk is answered with a read error naming the Shell route, with no client call", () => {
    for (const [range, lines] of [[{}, `1,${BOUND}`], [{ offset: 50, limit: 20 }, "50,69"]] as const) {
      const { sent, calls } = handled(readExec(9, ABSENT, range), tools("Read"));
      expect(calls).toEqual([]);
      expect(sent.length).toBe(1);
      const result = clientMessage(sent[0]!).value.message.value.result;
      expect(result.case).toBe("error");
      expect(result.value.error).toContain("no file is at that path on this machine");
      expect(result.value.error).toContain(`sed -n '${lines}p' '${ABSENT}' | cut -b 1-${BYTES}`);
      expect(result.value.error).toContain("Shell");
    }
  });

  test("a window past the scan bound is answered with a read error naming the Shell route, with no client call", () => {
    __test.limits.readScanBytes = 4096;   // the page's line 300 starts near byte 87,000
    const { sent, calls } = handled(readExec(9, PAGE, { offset: 300 }), tools("Read"));
    expect(calls).toEqual([]);
    expect(sent.length).toBe(1);
    const result = clientMessage(sent[0]!).value.message.value.result;
    expect(result.case).toBe("error");
    expect(result.value.error).toContain("past the first 4096 bytes the bridge measures");
    expect(result.value.error).toContain(`sed -n '300,${300 + BOUND - 1}p' '${PAGE}' | cut -b 1-${BYTES}`);
    // A window inside the scan bound still reads.
    expect(handled(readExec(10, SHORT), tools("Read")).args).toEqual([{ file_path: SHORT, limit: BOUND }]);
  });

  // F2: Read counts lines from the start only, so a negative offset is never
  // sent as the first lines: it is answered at once with a read error that
  // names tail, whatever the file's length, and nothing reads as its tail.
  test("a read from the end is answered with a read error naming tail, never the first lines as range_applied", () => {
    for (const path of [SHORT, LONG, ABSENT]) {
      const { sent, calls } = handled(readExec(9, path, { offset: -50 }), tools("Read"));
      expect(calls).toEqual([]);
      expect(sent.length).toBe(1);
      const result = clientMessage(sent[0]!).value.message.value.result;
      expect(result.case).toBe("error");
      expect(result.value.path).toBe(path);
      expect(result.value.error).toContain(`tail -n 50 '${path}'`);
      expect(result.value.error).toContain("Shell");
    }
    const some = handled(readExec(9, LONG, { offset: -50, limit: 10 }), tools("Read"));
    expect(clientMessage(some.sent[0]!).value.message.value.result.value.error).toContain(`tail -n 50 '${LONG}' | head -n 10`);
  });

  test("the note comes before text sent with the results, which still rides the read", async () => {
    const text = printed(LONG, 1, BOUND);
    const { sent } = await roundTrip(tools("Read"), [readExec(9, LONG)], [text], "stop and summarize");
    const content: string = resultOf(sent, "exec-9").value.result.value.output.value;
    expect(content.endsWith("\n\n<user_message>\nstop and summarize\n</user_message>")).toBe(true);
    expect(content.indexOf("sed -n")).toBeGreaterThan(text.length);
    expect(content.indexOf("sed -n")).toBeLessThan(content.indexOf("<user_message>"));
  });
});

// Regression test (2026-10-21): A GREP WITH NO GREP TOOL RUNS AS THE CALLER'S BASH
// RG, AND A REFUSAL NAMES THE TOOL AND WHY. MEASURED on a live session:
// Claude Code 2.1.306 registers no Grep tool, so Cursor's own grep was
// refused with "Cursor's own tool is refused in this environment"; in a
// ~135k-token conversation the model ended five turns in a row with no
// tool call, then told the user that native calls were refused. The bridge
// traced the same exec (grepArgs, no client call) with that session's real
// tool set. Nothing the bridge refused reached its log.
describe("a grep with no Grep tool runs as the caller's Bash rg", () => {
  const CLAUDE_CODE_TOOLS = () => tools("Bash", "Read", "Write", "Monitor");

  test("it reaches the client as one bounded rg command with its pattern, path and glob quoted", () => {
    const e = exec(20, "grepArgs", create(GrepArgsSchema, {
      pattern: "it's LANDED", path: "/tmp/a b/out.txt", glob: "*.output", caseInsensitive: true, contextAfter: 1, headLimit: 20,
    }));
    const { sent, calls, args } = handled(e, CLAUDE_CODE_TOOLS());
    expect(sent.length).toBe(0);
    expect(calls.map((c) => c.toolName)).toEqual(["Bash"]);
    expect(args).toEqual([{ command:
      "timeout 120 rg --no-config --color=never --no-heading --with-filename -n -A 1 -i --glob '*.output' " +
      "-e 'it'\\''s LANDED' -- '/tmp/a b/out.txt' | head -n 20" }]);
    const unbounded = handled(exec(21, "grepArgs", create(GrepArgsSchema, { pattern: "x", outputMode: "files_with_matches" })), CLAUDE_CODE_TOOLS());
    expect(unbounded.args[0]!.command).toBe("timeout 120 rg --no-config --color=never --no-heading --with-filename -l -e 'x' -- '.' | head -n 250");
  });

  test("a Grep that is registered still runs as Grep", () => {
    expect(handled(exec(22, "grepArgs", create(GrepArgsSchema, { pattern: "x" })), tools("Grep", "Bash")).calls.map((c) => c.toolName)).toEqual(["Grep"]);
  });

  test("rg's lines reach Cursor as the grep's success, split by file with line numbers", async () => {
    const { sent } = await roundTrip(CLAUDE_CODE_TOOLS(), [exec(23, "grepArgs", create(GrepArgsSchema, { pattern: "TODO", path: "src", context: 1 }))],
      ["src/a.ts-2-before\nsrc/a.ts:3:// TODO one\n--\nsrc/b.ts:10:x TODO\n"]);
    const grep = resultOf(sent, "exec-23").value.result;
    expect(grep.case).toBe("success");
    const content = (Object.values(grep.value.workspaceResults)[0] as any).result.value;
    expect(content.matches.map((f: any) => [f.file, f.matches.map((m: any) => [m.lineNumber, m.content, m.isContextLine])])).toEqual([
      ["src/a.ts", [[2, "before", true], [3, "// TODO one", false]]],
      ["src/b.ts", [[10, "x TODO", false]]],
    ]);
  });

  test("no match, a files list and counts are the grep's success too", async () => {
    const none = await roundTrip(CLAUDE_CODE_TOOLS(), [exec(24, "grepArgs", create(GrepArgsSchema, { pattern: "zzz" }))], ["(Bash completed with no output)"]);
    const empty = resultOf(none.sent, "exec-24").value.result;
    expect(empty.case).toBe("success");
    expect((Object.values(empty.value.workspaceResults)[0] as any).result.value.matches).toEqual([]);
    const files = await roundTrip(CLAUDE_CODE_TOOLS(), [exec(25, "grepArgs", create(GrepArgsSchema, { pattern: "x", outputMode: "files_with_matches" }))],
      ["src/a.ts\nsrc/b.ts\n"]);
    const f = (Object.values(resultOf(files.sent, "exec-25").value.result.value.workspaceResults)[0] as any).result;
    expect([f.case, f.value.files, f.value.totalFiles]).toEqual(["files", ["src/a.ts", "src/b.ts"], 2]);
    const counts = await roundTrip(CLAUDE_CODE_TOOLS(), [exec(26, "grepArgs", create(GrepArgsSchema, { pattern: "x", outputMode: "count" }))],
      ["src/a.ts:2\nsrc/b.ts:1"]);
    const c = (Object.values(resultOf(counts.sent, "exec-26").value.result.value.workspaceResults)[0] as any).result;
    expect([c.case, c.value.totalFiles, c.value.totalMatches]).toEqual(["count", 2, 3]);
  });

  test("an rg error, a non-zero exit and a refused Bash are the grep's error carrying the words", async () => {
    const answers = ["rg: /nope: No such file or directory (os error 2)", "Exit code 2\nrg: regex parse error", "<tool_use_error>denied by hook</tool_use_error>"];
    for (const [i, out] of answers.entries()) {
      const { sent } = await roundTrip(CLAUDE_CODE_TOOLS(), [exec(30 + i, "grepArgs", create(GrepArgsSchema, { pattern: "(" }))], [out]);
      const grep = resultOf(sent, `exec-${30 + i}`).value.result;
      expect(grep.case).toBe("error");
      expect(grep.value.error).toContain(i === 2 ? "denied by hook" : out);
    }
  });

  test("a native the bridge refuses is answered with the tool and why, and the bridge log says so", () => {
    const log = spyOn(console, "error").mockImplementation(() => {});
    try {
      const del = reply(exec(40, "deleteArgs", create(DeleteArgsSchema, { path: "/tmp/x" })), CLAUDE_CODE_TOOLS());
      expect(del.message.value.result.value.reason).toBe(refusal("Delete", "a delete has no caller tool that runs it", BASH_ROUTE));
      expect(log.mock.calls.map((c) => String(c[0]))).toContain(
        `[bridge] native deleteArgs refused: ${refusal("Delete", "a delete has no caller tool that runs it", BASH_ROUTE)}`);
      const grep = reply(exec(41, "grepArgs", create(GrepArgsSchema, { pattern: "x" })), tools("Monitor"));
      expect(grep.message.value.result.value.error).not.toContain("refused in this environment");
    } finally {
      log.mockRestore();
    }
  });
});
