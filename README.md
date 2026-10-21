# Cursor OpenAI API

A standalone CLI that serves Cursor's API as an OpenAI-compatible endpoint.

This is a fork of [egoist/cursor-openai-api](https://github.com/egoist/cursor-openai-api).
It keeps the same CLI and endpoints and makes the server hold up under a long
agent session, such as Claude Code talking to it through
[CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI). See
[What this fork adds](#what-this-fork-adds).

## Sponsor

[ChatWise](https://chatwise.app?ref=cursor-openai-api) - All-in-one agentic chat app.

## Features

- OAuth authentication with Cursor
- OpenAI-compatible `/v1/chat/completions` endpoint
- Automatic model discovery from your Cursor account
- Tool calling support
- Streaming responses

## NPM

The npm package and the Docker image below are upstream's releases. To run
this fork, clone it and follow [Development](#development).

```bash
npm i -g cursor-openai-api

cursor-openai-api login
cursor-openai-api serve
```

## Docker

```bash
# Pull the image
docker pull ghcr.io/egoist/cursor-openai-api:latest

# Run
docker run -p 3000:3000 \
  -v ~/.config/cursor-openai-api:/home/appuser/.config/cursor-openai-api \
  ghcr.io/egoist/cursor-openai-api:latest login
docker run -p 3000:3000 \
  -v ~/.config/cursor-openai-api:/home/appuser/.config/cursor-openai-api \
  ghcr.io/egoist/cursor-openai-api:latest serve
```

Credentials are stored in `~/.config/cursor-openai-api/credentials.json`.

The image above is upstream's. This fork's server listens on `127.0.0.1` only,
so a container built from this fork is reachable through `-p` only if you
change that bind; on Linux, `--network host` works as is.

## Commands

### `cursor-openai-api login`

Authenticate with Cursor via OAuth browser flow.

```bash
cursor-openai-api login
```

### `cursor-openai-api logout`

Clear stored credentials.

```bash
cursor-openai-api logout
```

### `cursor-openai-api whoami`

Check authentication status.

```bash
cursor-openai-api whoami
```

### `cursor-openai-api models`

List available Cursor models.

```bash
cursor-openai-api models
```

### `cursor-openai-api serve`

Start the OpenAI-compatible proxy server on `127.0.0.1`. The port is the first
argument, else the `PORT` environment variable, else 3000. A port that is not
a number from 1 to 65535 is an error.

```bash
cursor-openai-api serve 3000
PORT=3000 cursor-openai-api serve
```

The server exposes:

- `POST /v1/chat/completions` - Chat completions endpoint
- `GET /v1/models` - List available models

### `cursor-openai-api status`

Check if the proxy server is running.

```bash
cursor-openai-api status
```

## Usage with OpenAI SDK

```javascript
import OpenAI from "openai"

const client = new OpenAI({
  apiKey: "cursor", // any non-empty string
  baseURL: "http://localhost:3000/v1",
})

const chat = await client.chat.completions.create({
  model: "claude-4-sonnet",
  messages: [{ role: "user", content: "Hello!" }],
})
```

## Requirements

- Bun to run the server, and Node.js 18+ on `PATH` (each Cursor run streams
  through a small Node HTTP/2 child process)
- `curl` on `PATH` (model discovery)
- macOS, Linux, or WSL
- A Cursor account with API access

## Environment variables

| Variable | Default | What it does |
| --- | --- | --- |
| `PORT` | `3000` | Port for `serve` when no port argument is given. |
| `HOME` | (set by your shell) | Credentials live in `$HOME/.config/cursor-openai-api/`. |
| `CURSOR_BRIDGE_TRACE` | unset (off) | Set to `1` to log every frame Cursor sends to stderr, with a timestamp, including the field numbers of messages the bundled proto does not know. |
| `CURSOR_BRIDGE_STALL_MS` | `90000` | Stall window in milliseconds: a run is ended when Cursor sends nothing at all for this long, or only heartbeats while one of its tool calls is open. Held between `30000` and `225000` with a warning; a value that is not a whole number keeps the default with a warning. |
| `CURSOR_BRIDGE_TOOL_UNAVAILABLE` | `tool not available` | Phrase (matched case-insensitively) that marks a run whose tools Cursor reported unavailable. Such a run's conversation is not kept, so the next request replays the history. Change it if Cursor rewords the message. |

## What this fork adds

Every change below has a regression test in `test/`.

**Serving**

- `serve` refreshes the access token from the stored refresh token when it is
  close to expiry, so a long-running server does not stop working when the
  token expires.
- `serve <port>` uses the port argument; upstream read only `$PORT`.
- The server listens on `127.0.0.1` only. It takes no credential from its
  callers and spends your Cursor account, so it is not exposed on the network.
- `/v1/models` answers `502` when model discovery fails, instead of a fixed
  list of made-up models. The Cursor token reaches `curl` on stdin, never on
  its command line, where other local users could read it.

**Conversations**

- Message content sent as an array of blocks (as Claude Code does) is read as
  text, not as `[object Object]`.
- Earlier turns reach Cursor as structured history, with tool calls and tool
  results in order.
- When a client turn only extends the previous one, the server continues
  Cursor's own conversation (its checkpoint and conversation id) and sends just
  the new user text. A compaction, `/clear`, an edited history or a run that
  did not end cleanly replays the full history instead, and the log says why.
- A paused run is found by the ids of the tool calls it handed out, so a main
  conversation and its subagents never resume each other's runs. A paused run
  nobody resumes is ended after 15 minutes.

**Tools**

- When Cursor wants to use one of its own tools (read, shell, grep, write, ls,
  fetch) and the client registered an equivalent (`Read`, `Bash`, `Grep`,
  `Write`, `WebFetch`), the call goes to the client as that tool's call, and
  the output goes back to Cursor in the shape Cursor expects. Grep falls back
  to one bounded `rg` command on `Bash` when the client has no `Grep`. Tools
  with no equivalent are refused with a message that names the route the model
  can use instead.
- A read brings at most 400 lines and 24,000 bytes of a file into the client's
  context; when the bound cuts a read, Cursor is told where it stopped and how
  to read the rest.
- Cursor's MCP state query is answered, and tool-call ids are made safe for
  Claude Code's id alphabet and still match on the way back.

**Errors and limits**

- Reasoning is sent as `reasoning_content`, separate from the answer.
- A Cursor quota or availability error becomes an HTTP `429` with
  `Retry-After` (or a stream error the client retries), never reply text.
- No fixed two-minute limit on a run. A run ends when Cursor's stream dies,
  when Cursor is stuck on an open tool call (see `CURSOR_BRIDGE_STALL_MS`), or
  just before the client's idle timeout, and the client is told why.
- A client that disconnects mid-stream no longer crashes the server.

**Usage**

- Every response reports `usage`. Both counts are estimates (UTF-8 bytes over
  four) of what the request and the response carried, because Cursor's own
  token counts describe Cursor's internal conversation, not the one the client
  sent. The `usage_source: "estimate"` field says so.

## Development

```bash
# Install dependencies
bun install

# Log in, then run without building
bun run src/cli.ts login
bun run src/cli.ts serve 3000

# Build for production
bun run build
```

## Tests

```bash
bun test
```

The tests drive the server with a scripted Cursor stream, so they need no
Cursor account and no internet access; they do need Node.js on `PATH`.
To type-check:

```bash
bun x tsc --noEmit
```

## Publishing

Push a tag to trigger the Docker build:

```bash
git tag v0.0.1
git push origin v0.0.1
```

## Prior art

This fork is based on [egoist/cursor-openai-api](https://github.com/egoist/cursor-openai-api)
by EGOIST, which is modified from https://github.com/ephraimduncan/opencode-cursor.
The structured-history shapes follow
[edgebyte-ai/cliproxyapi-cursor-provider-plugin](https://github.com/edgebyte-ai/cliproxyapi-cursor-provider-plugin) (MIT).

## License

MIT, as upstream. See [LICENSE](LICENSE).
