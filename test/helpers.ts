// Shared fakes for the bridge tests: a scripted h2 child and Cursor frames.
import { create, toBinary } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { fromJson } from "@bufbuild/protobuf";
import {
  AgentServerMessageSchema,
  ExecServerMessageSchema,
  InteractionUpdateSchema,
  McpArgsSchema,
  TextDeltaUpdateSchema,
  ThinkingDeltaUpdateSchema,
  TokenDeltaUpdateSchema,
  HeartbeatUpdateSchema,
  ConversationStateStructureSchema,
  ConversationTokenDetailsSchema,
  type AgentServerMessage,
} from "../src/proto/agent_pb";

export function frame(msg: AgentServerMessage, flags = 0): Buffer {
  const body = toBinary(AgentServerMessageSchema, msg);
  const head = Buffer.alloc(5);
  head[0] = flags;
  head.writeUInt32BE(body.length, 1);
  return Buffer.concat([head, Buffer.from(body)]);
}

export function mcpCall(execMsgId: number, toolCallId: string, command: string): AgentServerMessage {
  return create(AgentServerMessageSchema, {
    message: {
      case: "execServerMessage",
      value: create(ExecServerMessageSchema, {
        id: execMsgId,
        execId: `exec-${execMsgId}`,
        message: {
          case: "mcpArgs",
          value: create(McpArgsSchema, {
            name: "Bash",
            toolName: "Bash",
            toolCallId,
            providerIdentifier: "opencode",
            args: { command: toBinary(ValueSchema, fromJson(ValueSchema, command)) },
          }),
        },
      }),
    },
  });
}

export function textDelta(text: string, thinking = false): AgentServerMessage {
  return create(AgentServerMessageSchema, {
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: thinking
          ? { case: "thinkingDelta", value: create(ThinkingDeltaUpdateSchema, { text }) }
          : { case: "textDelta", value: create(TextDeltaUpdateSchema, { text }) },
      }),
    },
  });
}

export function heartbeat(): AgentServerMessage {
  return create(AgentServerMessageSchema, {
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: { case: "heartbeat", value: create(HeartbeatUpdateSchema, {}) },
      }),
    },
  });
}

export function tokenDelta(tokens: number): AgentServerMessage {
  return create(AgentServerMessageSchema, {
    message: {
      case: "interactionUpdate",
      value: create(InteractionUpdateSchema, {
        message: { case: "tokenDelta", value: create(TokenDeltaUpdateSchema, { tokens }) },
      }),
    },
  });
}

export function checkpoint(usedTokens: number): AgentServerMessage {
  return create(AgentServerMessageSchema, {
    message: {
      case: "conversationCheckpointUpdate",
      value: create(ConversationStateStructureSchema, {
        tokenDetails: create(ConversationTokenDetailsSchema, { usedTokens, maxTokens: 256_000 }),
      }),
    },
  });
}

/** A stand-in for spawnBridge(): records writes, lets the test push bytes and close. */
export function fakeBridge() {
  let onData: ((c: Buffer) => void) | null = null;
  let onClose: ((code: number) => void) | null = null;
  const writes: Uint8Array[] = [];
  const state = { killed: false, ended: false };
  return {
    writes,
    state,
    bridge: {
      proc: { kill: () => { state.killed = true; onClose?.(143); } } as never,
      write: (d: Uint8Array) => { writes.push(d); },
      end: () => { state.ended = true; },
      onData: (cb: (c: Buffer) => void) => { onData = cb; },
      onClose: (cb: (code: number) => void) => { onClose = cb; },
    },
    feed: (...frames: Buffer[]) => onData?.(Buffer.concat(frames)),
    close: (code = 0) => onClose?.(code),
  };
}

/** A bare run over a fake bridge, as startRun would build it. */
export function makeRun(bridge: ReturnType<typeof fakeBridge>["bridge"], id = 1) {
  const timer = setInterval(() => {}, 60_000);
  clearInterval(timer);
  return {
    id,
    modelId: "grok-4.7-high",
    bridge,
    heartbeatTimer: timer,
    blobStore: new Map(),
    mcpTools: [],
    awaiting: [],
    queued: [],
    held: [],
    sink: null,
    pauseAfterChunk: false,
    pending: Buffer.alloc(0),
    closed: false,
    expiry: null,
    stall: null,
    contextTokens: 0,
    heldOutputTokens: 0,
  } as never;
}

export type Sse = {
  content: string;
  reasoning: string;
  toolCalls: Array<{ id: string; name: string; arguments: string }>;
  finish: string | null;
  usage: Record<string, unknown> | null;
};

export async function readSse(resp: Response): Promise<Sse> {
  const out: Sse = { content: "", reasoning: "", toolCalls: [], finish: null, usage: null };
  const text = await resp.text();
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
    const c = JSON.parse(line.slice(6));
    if (c.usage) out.usage = c.usage;
    const ch = c.choices[0];
    out.content += ch.delta.content ?? "";
    out.reasoning += ch.delta.reasoning_content ?? "";
    for (const tc of ch.delta.tool_calls ?? []) {
      out.toolCalls.push({ id: tc.id, name: tc.function.name, arguments: tc.function.arguments });
    }
    if (ch.finish_reason) out.finish = ch.finish_reason;
  }
  return out;
}
