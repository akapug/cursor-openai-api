/**
 * Cursor OpenAI API - Standalone CLI
 *
 * Serves Cursor API as OpenAI-compatible endpoint with OAuth authentication.
 */
import { login, logout, getStoredCredentials, isAuthenticated, refreshCursorToken, saveCredentials } from "./cli-auth";
import { startProxy, stopProxy, getProxyPort } from "./proxy";
import { getCursorModels } from "./models";
import { getTokenExpiry } from "./auth";

const PROXY_PORT = 3000;

async function cmdLogin() {
  console.log("🔐 Starting Cursor OAuth login...\n");
  try {
    const result = await login();
    console.log("\n✅ Login successful!");
    console.log(`   Access token: ${result.accessToken.slice(0, 20)}...`);
    console.log(`   Expires in: ${Math.round((getTokenExpiry(result.accessToken) - Date.now()) / 1000 / 60)} minutes`);
  } catch (error) {
    console.error("\n❌ Login failed:", error);
    process.exit(1);
  }
}

function cmdLogout() {
  logout();
  console.log("👋 Logged out successfully.");
}

async function cmdWhoami() {
  const creds = getStoredCredentials();
  if (!creds) {
    console.log("❌ Not logged in. Run `cursor-api login` to authenticate.");
    return;
  }

  const expiresIn = Math.round((creds.expires - Date.now()) / 1000 / 60);
  if (expiresIn < 0) {
    console.log("⚠️  Token expired. Run `cursor-api login` to re-authenticate.");
    return;
  }

  console.log("✅ Authenticated with Cursor");
  console.log(`   Token expires in: ${expiresIn} minutes`);
}

async function cmdModels() {
  const creds = getStoredCredentials();
  if (!creds) {
    console.log("❌ Not logged in. Run `cursor-api login` first.");
    process.exit(1);
  }

  console.log("🔍 Fetching available models...\n");
  try {
    const models = await getCursorModels(creds.access);
    console.log("Available Cursor models:\n");
    for (const model of models) {
      const reasoning = model.reasoning ? "🧠" : "   ";
      console.log(`  ${reasoning} ${model.id.padEnd(25)} ${model.name}`);
    }
  } catch (error) {
    console.error("Failed to fetch models:", error);
    process.exit(1);
  }
}

async function cmdServe(port: number) {
  // helm patch (2026-08-11): upstream cmdServe captured `creds.access` ONCE,
  // so a long-running serve outlived its access token and every request then
  // failed until a manual restart — the refresh helper existed but nothing in
  // the serve path called it. Re-read the store per request and refresh
  // through the stored refresh token when the access token is within 60s of
  // expiry (or past it): one owner login, then the bridge keeps itself live.
  const getAccessToken = async (): Promise<string> => {
    const c = getStoredCredentials();
    if (!c) throw new Error("Not logged in. Run `cursor-api login` first.");
    if (c.expires > Date.now() + 60_000) return c.access;
    const fresh = await refreshCursorToken(c.refresh);
    saveCredentials(fresh);
    return fresh.access;
  };

  const creds = getStoredCredentials();
  if (!creds) {
    console.log("❌ Not logged in. Run `cursor-api login` first.");
    process.exit(1);
  }

  if (creds.expires < Date.now()) {
    try {
      await getAccessToken();
      console.log("🔄 Stored token was expired; refreshed via refresh token.");
    } catch {
      console.log("⚠️  Token expired and refresh failed. Run `cursor-api login` to re-authenticate.");
      process.exit(1);
    }
  }

  console.log(`🚀 Starting Cursor OpenAI API proxy on port ${port}...`);
  console.log(`   Endpoint: http://localhost:${port}/v1`);
  console.log(`   Models: http://localhost:${port}/v1/models`);
  console.log(`   Press Ctrl+C to stop.\n`);

  const proxyPort = await startProxy(getAccessToken, port);

  console.log(`✅ Server running at http://localhost:${proxyPort}\n`);

  process.on("SIGINT", () => {
    console.log("\n\n👋 Shutting down...");
    stopProxy();
    process.exit(0);
  });
}

function cmdStatus() {
  const port = getProxyPort();
  if (port) {
    console.log(`🟢 Proxy running on port ${port}`);
  } else {
    console.log("🔴 Proxy not running");
  }
}

function showHelp() {
  console.log(`
cursor-api - Cursor OpenAI API CLI

Usage: cursor-api <command> [options]

Commands:
  login                 Authenticate with Cursor via OAuth
  logout                Clear stored credentials
  whoami                Show current authentication status
  models                List available Cursor models
  serve [port]          Start the OpenAI-compatible proxy server (default port: ${PROXY_PORT})
  status                Show proxy server status
  help                  Show this help message

Examples:
  cursor-api login
  cursor-api models
  cursor-api serve 8080
`);
}

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] ?? "help";

  switch (command) {
    case "login":
      await cmdLogin();
      break;
    case "logout":
      cmdLogout();
      break;
    case "whoami":
      await cmdWhoami();
      break;
    case "models":
      await cmdModels();
      break;
    case "serve": {
      const envPort = parseInt(process.env.PORT ?? "", 10);
      const port = envPort || PROXY_PORT;
      if (isNaN(port) || port < 1 || port > 65535) {
        console.error(`Invalid port: ${process.env.PORT}. Using default ${PROXY_PORT}.`);
      }
      await cmdServe(port);
      break;
    }
    case "status":
      cmdStatus();
      break;
    case "help":
    case "--help":
    case "-h":
    default:
      showHelp();
      break;
  }
}

main();
