// Live preview for writing: re-renders the page whenever pages/, images/, data/ or the source
// config change, serves it on localhost with the images, and the browser reloads itself.
// Usage: npm run dev   (then open http://127.0.0.1:8000/)
// Edits to build.mjs itself (theme, layout, metadata) need a restart, since the module is loaded once.
import { existsSync, readFileSync, statSync, watch } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { render } from "./build.mjs";

const root = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 8000);
const TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".svg": "image/svg+xml", ".webp": "image/webp", ".txt": "text/plain", ".xml": "application/xml" };

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

// Images come straight from images/ (the page references them as images/...); the favicon too.
function staticFile(url) {
  const path = decodeURIComponent(url.split("?")[0]);
  if (path === "/favicon.png") return join(root, "images", "favicon.png");
  if (path.startsWith("/images/")) {
    const file = normalize(join(root, path));
    return file.startsWith(join(root, "images") + "/") ? file : "";   // "" = refuse (outside images/)
  }
  return null;
}

rebuild();
createServer((req, res) => {
  if (req.url === "/__version") {
    res.writeHead(200, { "Content-Type": "text/plain", "Cache-Control": "no-store" });
    return res.end(String(version));
  }
  const file = staticFile(req.url);
  if (file !== null) {
    if (!file || !existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": TYPES[extname(file).toLowerCase()] || "application/octet-stream", "Cache-Control": "no-store" });
    return res.end(readFileSync(file));
  }
  res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
  res.end(html);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`live preview at http://127.0.0.1:${PORT}/  (watching ${watched.filter(existsSync).length} paths)`);
});
