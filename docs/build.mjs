// Build the writeup: index.md -> HTML, inline images, encrypt with StatiCrypt into site/.
// Usage: npm run build
// Passphrase: STATICRYPT_PASSWORD from docs/.env (gitignored) or the environment; prompts if unset.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const root = dirname(fileURLToPath(import.meta.url));
const TITLE = "VLA Test";
const FAVICON = join(root, "images", "favicon.png"); // small square PNG; see README to regenerate

const dataUri = (file, type) => `data:${type};base64,${readFileSync(file).toString("base64")}`;
const faviconTag = `<link rel="icon" type="image/png" href="${dataUri(FAVICON, "image/png")}">`;

let body = marked.parse(readFileSync(join(root, "index.md"), "utf8"));

// Inline local images as data URIs so nothing is fetchable outside the encrypted page.
const mime = {
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
};
body = body.replace(/<img([^>]*?)src="([^"]+)"/g, (tag, attrs, src) => {
  if (/^(https?:|data:)/.test(src)) return tag;
  const file = join(root, src);
  const type = mime[extname(file).toLowerCase()];
  if (!type || !existsSync(file)) throw new Error(`image not found or unsupported: ${src}`);
  return `<img${attrs}src="${dataUri(file, type)}"`;
});

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${TITLE}</title>
${faviconTag}
<style>
  body { max-width: 46rem; margin: 3rem auto; padding: 0 1rem; font: 17px/1.6 system-ui, sans-serif; color: #222; background: #fff; }
  img { max-width: 100%; height: auto; }
  pre { overflow-x: auto; padding: 1rem; background: #f4f4f4; }
  code { font-size: 0.9em; }
  @media (prefers-color-scheme: dark) { body { color: #ddd; background: #111; } pre { background: #1c1c1c; } }
</style>
</head>
<body>
${body}
</body>
</html>
`;

const build = join(root, "build");
mkdirSync(build, { recursive: true });
writeFileSync(join(build, "index.html"), html);

// Keep the encrypted page out of search engines (Cloudflare Pages reads _headers).
mkdirSync(join(root, "site"), { recursive: true });
writeFileSync(join(root, "site", "_headers"), "/*\n  X-Robots-Tag: noindex, nofollow\n");

// Encrypt build/index.html -> site/index.html. The salt lives in .staticrypt.json; commit it.
execFileSync(
  join(root, "node_modules", ".bin", "staticrypt"),
  ["index.html", "-d", "../site", "--config", "../.staticrypt.json", "--remember", "30", "--template-title", TITLE],
  { cwd: build, stdio: "inherit" },
);

// The passphrase prompt is StatiCrypt's own template; give it the same favicon as the page behind it.
const out = join(root, "site", "index.html");
writeFileSync(out, readFileSync(out, "utf8").replace("</title>", `</title>\n        ${faviconTag}`));
