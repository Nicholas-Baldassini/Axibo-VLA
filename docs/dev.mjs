// Live preview for writing: re-renders the page whenever pages/, images/, data/ or the source
// config change, serves the unencrypted page on localhost, and the browser reloads itself.
// Usage: npm run dev   (then open http://127.0.0.1:8000/)
// Edits to build.mjs itself (theme, layout) need a restart, since the module is loaded once.
import { existsSync, watch } from "node:fs";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { render } from "./build.mjs";

const root = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8000);

let html = "";
let version = 0;
let timer = null;

function rebuild() {
  try {
    html = render({ dev: true });
    version += 1;
    console.log(`[${new Date().toLocaleTimeString()}] rendered (build ${version})`);
  } catch (err) {
    console.error(`[${new Date().toLocaleTimeString()}] render failed: ${err.message}`);
  }
}

const watched = [
  join(root, "pages"),
  join(root, "images"),
  join(root, "data"),
  join(root, "..", "source", "config.py"),
];
for (const target of watched) {
  if (!existsSync(target)) continue;
  watch(target, { recursive: true }, () => {
    clearTimeout(timer);
    timer = setTimeout(rebuild, 150);   // editors write in bursts; rebuild once per burst
  });
}

rebuild();
createServer((req, res) => {
  if (req.url === "/__version") {
    res.writeHead(200, { "Content-Type": "text/plain", "Cache-Control": "no-store" });
    return res.end(String(version));
  }
  res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  res.end(html);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`live preview at http://127.0.0.1:${PORT}/  (unencrypted; watching ${watched.filter(existsSync).length} paths)`);
});
