// helm arm (2026-09-25): a paused Cursor run is found by the tool-call ids it
// handed out, never by what the conversation looks like. The old key (model +
// first 200 chars of the first user message) was shared by a Claude Code seat
// and every subagent it spawned, so their tool round trips crossed.
import { afterEach, describe, expect, test } from "bun:test";
import { fromBinary } from "@bufbuild/protobuf";
import { AgentClientMessageSchema, type McpResult } from "../src/proto/agent_pb";
import { __test } from "../src/proxy";
import { fakeBridge, frame, makeRun, mcpCall, readSse, textDelta } from "./helpers";

const NO_PEEK = { buf: Buffer.alloc(0), refusal: null, closedCode: null };
// Claude Code opens the seat's conversation and every subagent's the same way.
const OPENING = { role: "user" as const, content: "<system-reminder>same preamble for everyone</system-reminder>" };

function started(id: number) {
  const fb = fakeBridge();
  const run = makeRun(fb.bridge, id);
  const resp = __test.respond(run, true, () => __test.listen(run, NO_PEEK)) as Response;
  return { fb, run, resp };
}

function asked(toolCallId: string) {
  return [
    OPENING,
    { role: "assistant" as const, content: null, tool_calls: [{ id: toolCallId, type: "function" as const, function: { name: "Bash", arguments: "{}" } }] },
    { role: "tool" as const, content: "ok", tool_call_id: toolCallId },
  ];
}

function resultsWritten(writes: Uint8Array[]): McpResult[] {
  return writes.map((w) => {
    const exec = fromBinary(AgentClientMessageSchema, w.subarray(5)).message.value as { message: { value: McpResult } };
    return exec.message.value;
  });
}

afterEach(() => __test.pausedRuns.clear());

describe("paused runs", () => {
  test("two conversations with the same opening each get their own run back", async () => {
    const a = started(1);
    const b = started(2);
    a.fb.feed(frame(mcpCall(1, "call-A\nfc_A", "echo A")));
    b.fb.feed(frame(mcpCall(1, "call-B\nfc_B", "echo B")));
    const [sa, sb] = [await readSse(a.resp), await readSse(b.resp)];
    expect(sa.finish).toBe("tool_calls");
    expect(sb.finish).toBe("tool_calls");
    expect(__test.findPausedRun(asked(sa.toolCalls[0]!.id))).toBe(a.run);
    expect(__test.findPausedRun(asked(sb.toolCalls[0]!.id))).toBe(b.run);

    // A's result goes to A's bridge only, and B stays paused.
    const resp = __test.resumeRun(a.run, [{ toolCallId: sa.toolCalls[0]!.id, content: "A-OUT" }], "", true) as Response;
    expect(resp).not.toBeNull();
    expect(resultsWritten(a.fb.writes).map((r) => r.result.case)).toEqual(["success"]);
    expect(b.fb.writes.length).toBe(0);
    expect(__test.findPausedRun(asked(sb.toolCalls[0]!.id))).toBe(b.run);
    expect(b.fb.state.killed).toBe(false);
    a.fb.feed(frame(textDelta("A saw A-OUT")));
    a.fb.close();
    expect((await readSse(resp)).content).toBe("A saw A-OUT");
  });

  test("tool messages before the last assistant message never resume a run", async () => {
    const a = started(1);
    a.fb.feed(frame(mcpCall(1, "call-A", "echo A")));
    const id = (await readSse(a.resp)).toolCalls[0]!.id;
    const later = [...asked(id), { role: "assistant" as const, content: "done" }, { role: "user" as const, content: "next" }];
    expect(__test.findPausedRun(later)).toBeNull();
  });

  test("a run that dies while paused leaves the map, and its results go stateless", async () => {
    const a = started(1);
    a.fb.feed(frame(mcpCall(1, "call-A", "echo A")));
    const id = (await readSse(a.resp)).toolCalls[0]!.id;
    a.fb.close(1);
    expect(__test.findPausedRun(asked(id))).toBeNull();
    expect(__test.resumeRun(a.run, [{ toolCallId: id, content: "x" }], "", true)).toBeNull();
  });

  test("parallel tool calls in one chunk reach the client together", async () => {
    const a = started(1);
    a.fb.feed(frame(textDelta("running both", false)), frame(mcpCall(1, "call-1", "echo 1")), frame(mcpCall(2, "call-2", "echo 2")));
    const s = await readSse(a.resp);
    expect(s.toolCalls.map((t) => t.id)).toEqual(["call-1", "call-2"]);
    expect(s.finish).toBe("tool_calls");
    expect(__test.pausedRuns.get("call-1")).toBe(a.run);
    expect(__test.pausedRuns.get("call-2")).toBe(a.run);
  });

  test("a tool call Cursor asks for while paused is handed out on resume", async () => {
    const a = started(1);
    a.fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
    await readSse(a.resp);
    a.fb.feed(frame(mcpCall(2, "call-2", "echo 2")));  // no client attached
    const resp = __test.resumeRun(a.run, [{ toolCallId: "call-1", content: "one" }], "", true) as Response;
    const s = await readSse(resp);
    expect(s.toolCalls.map((t) => t.id)).toEqual(["call-2"]);
    expect(s.finish).toBe("tool_calls");
    expect(__test.pausedRuns.get("call-2")).toBe(a.run);
    expect(__test.pausedRuns.has("call-1")).toBe(false);
  });

  test("text sent alongside the tool results rides the last result", async () => {
    const a = started(1);
    a.fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
    await readSse(a.resp);
    __test.resumeRun(a.run, [{ toolCallId: "call-1", content: "one" }], "stop and summarize", true);
    const [r] = resultsWritten(a.fb.writes);
    const text = (r!.result.value as { content: { content: { value: { text: string } } }[] }).content[0]!.content.value.text;
    expect(text).toContain("one");
    expect(text).toContain("<user_message>\nstop and summarize\n</user_message>");
  });

  test("a client that hangs up mid-answer ends the run instead of leaking it", async () => {
    const a = started(1);
    a.fb.feed(frame(textDelta("partial")));
    await a.resp.body!.cancel();
    expect(a.fb.state.killed).toBe(true);
  });

  test("a compacted conversation never resumes the run that holds the uncompacted one", async () => {
    // MEASURED on the live seat: Claude Code's autocompact keeps the last
    // tool call and its result after the summary, so the request after a
    // compaction answers the paused run's tool call. Resuming it kept Cursor
    // on the whole pre-compaction conversation: the compaction bought
    // nothing, every turn still billed ~145k input, and the seat compacted
    // three times in eight minutes.
    const a = started(1);
    const opened = [OPENING, { role: "user" as const, content: "do the task" }];
    __test.stampConversation(a.run, opened);
    a.fb.feed(frame(mcpCall(1, "call-A", "echo A")));
    const id = (await readSse(a.resp)).toolCalls[0]!.id;
    const [, call, result] = asked(id);
    const continued = [...opened, call!, result!];
    expect(__test.takePausedRun(continued)).toBe(a.run);
    __test.pausedRuns.set(id, a.run);
    const compacted = [{ role: "user" as const, content: "This session is being continued from a previous conversation. Summary: ..." }, call!, result!];
    expect(__test.takePausedRun(compacted)).toBeNull();
    expect(a.fb.state.killed).toBe(true);
    expect(a.fb.writes.length).toBe(0);
    expect(__test.findPausedRun(continued)).toBeNull();
  });

  test("a paused run arms its expiry, and resuming disarms it", async () => {
    const a = started(1);
    a.fb.feed(frame(mcpCall(1, "call-1", "echo 1")));
    await readSse(a.resp);
    expect((a.run as { expiry: unknown }).expiry).not.toBeNull();
    __test.resumeRun(a.run, [{ toolCallId: "call-1", content: "one" }], "", true);
    expect((a.run as { expiry: unknown }).expiry).toBeNull();
  });
});
