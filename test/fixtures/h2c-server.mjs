// A cleartext HTTP/2 stand-in for Cursor: accepts one stream, answers one
// chunk, then holds the stream open. Prints its port.
import http2 from "node:http2";
const server = http2.createServer();
server.on("stream", (stream) => {
  stream.respond({ ":status": 200, "content-type": "application/connect+proto" });
  stream.write(Buffer.from([0, 0, 0, 0, 0]));
  stream.on("data", () => {});
});
server.listen(0, "127.0.0.1", () => console.log(`port ${server.address().port}`));
