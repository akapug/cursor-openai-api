// helm arm (2026-09-25): prior turns reach Cursor as structured root-prompt
// messages — tool calls and tool results as typed parts, in order — instead
// of a text transcript that dropped every tool call and pasted every tool
// result at the top of the latest message.
import { describe, expect, test } from "bun:test";
import { fromBinary } from "@bufbuild/protobuf";
import { AgentClientMessageSchema, type AgentRunRequest } from "../src/proto/agent_pb";
import { __test } from "../src/proxy";

const CONVO = [
  { role: "system" as const, content: "be brief" },
  { role: "user" as const, content: [{ type: "text", text: "run the first one" }] as never },
  { role: "assistant" as const, content: "running it", tool_calls: [
    { id: "call-1", type: "function" as const, function: { name: "Bash", arguments: "{\"command\":\"echo FIRST-17\"}" } },
  ] },
  { role: "tool" as const, content: "FIRST-17\n", tool_call_id: "call-1" },
  { role: "assistant" as const, content: "It printed FIRST-17." },
  { role: "user" as const, content: "what did it print?" },
];

function rootMessages(req: ReturnType<typeof __test.buildCursorRequest>) {
  const msg = fromBinary(AgentClientMessageSchema, req.requestBytes);
  const run = msg.message.value as AgentRunRequest;
  const roots = run.conversationState!.rootPromptMessagesJson.map((id) =>
    JSON.parse(new TextDecoder().decode(req.blobStore.get(Buffer.from(id).toString("hex"))!)));
  const action = run.action!.action.value as { userMessage: { text: string } };
  return { roots, text: action.userMessage.text, turns: run.conversationState!.turns.length };
}

describe("structured history", () => {
  test("tool calls and results reach Cursor as typed parts, in order, and the latest user message is the action", () => {
    const { history, userText } = __test.toHistory(CONVO);
    const { roots, text, turns } = rootMessages(__test.buildCursorRequest("grok-4.7-high", "be brief", history, userText));
    expect(text).toBe("what did it print?");
    expect(turns).toBe(0);
    expect(roots).toEqual([
      { role: "system", content: "be brief" },
      { role: "user", content: [{ type: "text", text: "run the first one" }] },
      { role: "assistant", content: [
        { type: "text", text: "running it" },
        { type: "tool-call", toolCallId: "call-1", toolName: "opencode-Bash", args: { command: "echo FIRST-17" } },
      ] },
      { role: "tool", id: "call-1", content: [
        { type: "tool-result", toolCallId: "call-1", toolName: "opencode-Bash", result: "FIRST-17\n" },
      ] },
      { role: "assistant", content: [{ type: "text", text: "It printed FIRST-17." }] },
    ]);
  });

  test("a request that ends in tool results asks Cursor to continue from them", () => {
    const { history, userText } = __test.toHistory(CONVO.slice(0, 4));
    expect(userText).toMatch(/^Continue from where you left off/);
    expect(history.at(-1)!.role).toBe("tool");
  });

  test("the text after the last assistant message is what rides a resumed run", () => {
    const parsed = __test.parseMessages([
      ...CONVO.slice(0, 3),
      { role: "tool" as const, content: "FIRST-17\n", tool_call_id: "call-1" },
      { role: "user" as const, content: "<system-reminder>r</system-reminder>" },
    ]);
    expect(parsed.tailText).toBe("<system-reminder>r</system-reminder>");
    expect(parsed.toolResults).toEqual([{ toolCallId: "call-1", content: "FIRST-17\n" }]);
    expect(parsed.systemPrompt).toBe("be brief");
  });
});
