// Build the writeup: pages/*.md -> one HTML page with a navbar, per-page table of contents and
// inlined images, then encrypt it with StatiCrypt into site/.
// Usage: npm run build          render + encrypt into site/ (what Vercel serves)
//        npm run dev            live preview: re-renders on save and reloads the browser, no passphrase
// Passphrase: STATICRYPT_PASSWORD from docs/.env (gitignored) or the environment; prompts if unset.
//
// A missing image renders as a labelled placeholder naming the file to add.
// Themes: "retro" (a 1990s desktop) and "modern" (dark) both ship in the page; a button at the top
// right switches between them and the choice is remembered in the browser. DEFAULT_THEME is the first visit.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Marked } from "marked";

const root = dirname(fileURLToPath(import.meta.url));
const TITLE = "VLA Test";
const DEFAULT_THEME = "retro";   // "retro" or "modern"
const FAVICON = join(root, "images", "favicon.png"); // small square PNG; see README to regenerate

const PAGES = [
  { id: "task1", title: "Task 1", file: "pages/task1.md" },
  { id: "task2", title: "Task 2", file: "pages/task2.md" },
  { id: "task3", title: "Task 3", file: "pages/task3.md" },
  { id: "task4", title: "Task 4", file: "pages/task4.md" },
];

const dataUri = (file, type) => `data:${type};base64,${readFileSync(file).toString("base64")}`;
const faviconTag = `<link rel="icon" type="image/png" href="${dataUri(FAVICON, "image/png")}">`;
const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const missing = [];

// Inline local images as data URIs so nothing is fetchable outside the encrypted page.
const mime = {
  ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
  ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
};
function inlineImages(html) {
  return html.replace(/<img\b([^>]*)>/g, (tag, attrs) => {
    const src = /\bsrc="([^"]+)"/.exec(attrs)?.[1];
    if (!src || /^(https?:|data:)/.test(src)) return tag;
    const alt = /\balt="([^"]*)"/.exec(attrs)?.[1] ?? "";
    const file = join(root, src);
    const type = mime[extname(file).toLowerCase()];
    if (!type) throw new Error(`unsupported image type: ${src}`);
    if (!existsSync(file)) {
      missing.push(src);
      return `<span class="img-placeholder"><strong>${escapeHtml(alt || "Image")}</strong>` +
        `Placeholder: add the file docs/${escapeHtml(src)}</span>`;
    }
    return `<img${attrs.replace(/\bsrc="[^"]+"/, `src="${dataUri(file, type)}"`)}>`;
  });
}

// Render one page: headings get ids prefixed with the page id and feed its table of contents.
function renderPage(page) {
  const toc = [];
  const marked = new Marked({
    renderer: {
      heading({ tokens, depth }) {
        const text = this.parser.parseInline(tokens);
        const id = `${page.id}-${slug(text)}`;
        if (depth === 2 || depth === 3) toc.push({ depth, id, text: text.replace(/<[^>]+>/g, "") });
        return `<h${depth} id="${id}">${text}</h${depth}>\n`;
      },
    },
  });
  const body = inlineImages(marked.parse(readFileSync(join(root, page.file), "utf8")));
  const tocHtml = toc.length
    ? `<ul>${toc.map((t) => `<li class="d${t.depth}"><a href="#${t.id}">${t.text}</a></li>`).join("")}</ul>`
    : `<p class="empty">No sections yet.</p>`;
  return {
    section: `<section class="page" data-id="${page.id}" hidden>\n${body}</section>`,
    toc: `<nav class="toc" data-page="${page.id}" hidden><div class="toc-title">Contents</div>${tocHtml}</nav>`,
    link: `<a href="#${page.id}" data-page="${page.id}">${page.title}</a>`,
  };
}

// In dev mode the page polls the dev server and reloads itself after every rebuild.
const LIVE_RELOAD = `<script>
(() => { let seen = null; setInterval(async () => {
  try { const v = await (await fetch("/__version", { cache: "no-store" })).text(); if (seen !== null && v !== seen) location.reload(); seen = v; } catch {}
}, 1000); })();
</script>`;

// Layout rules shared by both themes. Themes only set colours, fonts, borders and spacing.
const SHARED_CSS = `
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  img { max-width: 100%; height: auto; }
  .topbar { position: fixed; top: 0; left: 0; right: 0; display: flex; align-items: center; z-index: 10; }
  .topbar .spacer { flex: 1; }
  .theme-toggle { cursor: pointer; white-space: nowrap; }
  .win-buttons, .ticker, .statusbar { display: none; }
  .img-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .3rem; min-height: 12rem; }
  .toc ul { list-style: none; margin: 0; padding: 0; }
  .toc li a { display: block; text-decoration: none; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { text-align: left; vertical-align: top; }
  pre { overflow: auto; }
  .placeholder { font-style: italic; }
  @keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-100%); } }
  @keyframes blink { 50% { opacity: 0; } }
  @media (max-width: 900px) {
    .layout { grid-template-columns: minmax(0, 1fr) !important; gap: 1.5rem !important; }
    .sidebar { position: static !important; max-height: none !important; }
    .topbar { overflow-x: auto; }
  }
`;

// Modern: a dark reading theme.
const MODERN_CSS = `
html[data-theme="modern"] {
  color-scheme: dark;
  --bg: #0f1115; --fg: #e4e6eb; --muted: #9aa0a6; --line: #2a2f3a; --accent: #7aa2ff; --code: #181c24; --bar: #14171d;
  body { margin: 0; font: 17px/1.6 system-ui, sans-serif; color: var(--fg); background: var(--bg); }
  a { color: var(--accent); }
  .topbar { height: 3.25rem; gap: 1.5rem; padding: 0 1.25rem; background: var(--bar); border-bottom: 1px solid var(--line); }
  .topbar .brand { font-weight: 700; margin-right: .5rem; }
  .topbar nav a { color: var(--fg); text-decoration: none; padding: .35rem .7rem; border-radius: .4rem; }
  .topbar nav a.active { background: var(--accent); color: #fff; }
  .theme-toggle { font: inherit; font-size: .85rem; color: var(--fg); background: transparent; border: 1px solid var(--line); border-radius: .4rem; padding: .3rem .7rem; }
  .theme-toggle:hover { border-color: var(--accent); }
  .layout { display: grid; grid-template-columns: 16rem minmax(0, 46rem); gap: 3rem; max-width: 70rem; margin: 0 auto; padding: 4.75rem 1.25rem 4rem; }
  .sidebar { position: sticky; top: 4.5rem; align-self: start; max-height: calc(100vh - 5.5rem); overflow-y: auto; font-size: .92rem; }
  .toc-title { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin-bottom: .5rem; }
  .toc ul { border-left: 2px solid var(--line); }
  .toc li a { padding: .22rem .75rem; color: var(--muted); border-left: 2px solid transparent; margin-left: -2px; }
  .toc li.d3 a { padding-left: 1.6rem; font-size: .88rem; }
  .toc li a.active { color: var(--accent); border-left-color: var(--accent); }
  .toc li a:hover { color: var(--fg); }
  main h1 { font-size: 2rem; margin-top: 0; }
  main h2 { margin-top: 2.5rem; padding-top: .5rem; border-top: 1px solid var(--line); }
  main h2, main h3 { scroll-margin-top: 4.25rem; }
  img { border: 1px solid var(--line); border-radius: .3rem; }
  .img-placeholder { border: 2px dashed var(--line); border-radius: .3rem; color: var(--muted); font-size: .9rem; }
  pre { padding: 1rem; background: var(--code); border-radius: .3rem; font-size: .84rem; line-height: 1.45; }
  code { font-size: .9em; }
  :not(pre) > code { background: var(--code); padding: .1rem .3rem; border-radius: .25rem; }
  table { font-size: .92rem; }
  th, td { padding: .4rem .6rem; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 600; }
  mark { background: #fff3a3; color: #222; padding: 0 .2rem; }
  .placeholder { color: var(--muted); }
}
`;

// A 1990s desktop: teal wallpaper, blue-gradient title bars, bevelled windows, Times and Courier,
// a scrolling ticker, a status bar, and construction-tape placeholders.
const RETRO_CSS = `
html[data-theme="retro"] {
  color-scheme: light;
  --grey: #c0c0c0; --dark: #808080; --navy: #000080; --sky: #1084d0;
  --titlebar: linear-gradient(90deg, var(--navy), var(--sky));
  body { margin: 0; font: 16px/1.5 "Times New Roman", Times, serif; color: #000; background: #008080; }
  a { color: #0000ff; } a:visited { color: #800080; }
  a:hover { color: #ff0000; }
  ::selection { background: var(--navy); color: #fff; }

  .topbar { height: 2.75rem; gap: .35rem; padding: 0 .4rem 0 .6rem; background: var(--titlebar); border: 2px outset #dfdfdf; border-bottom-color: #404040; }
  .topbar .brand { font: bold 17px "Times New Roman", Times, serif; color: #fff; text-shadow: 1px 1px #000; margin-right: .6rem; }
  .topbar nav a { color: #000; text-decoration: none; padding: .2rem .8rem; background: var(--grey); border: 2px outset #fff; font: 13px Verdana, Tahoma, sans-serif; }
  .topbar nav a.active { border-style: inset; background: #a8a8a8; }
  .theme-toggle { font: 12px Verdana, Tahoma, sans-serif; color: #000; background: var(--grey); border: 2px outset #fff; padding: .2rem .6rem; }
  .theme-toggle:active { border-style: inset; }
  .win-buttons { display: flex; gap: 2px; margin-left: .4rem; }
  .win-buttons span { width: 18px; height: 16px; background: var(--grey); border: 2px outset #fff; font: bold 10px/12px Verdana, sans-serif; text-align: center; color: #000; }

  .ticker { display: block; position: fixed; top: 2.75rem; left: 0; right: 0; height: 1.5rem; overflow: hidden; z-index: 9;
            background: #000; color: #00ff00; font: 13px/1.5rem "Courier New", Courier, monospace; border-bottom: 2px solid var(--dark); white-space: nowrap; }
  .ticker span { display: inline-block; padding-left: 100%; animation: ticker 60s linear infinite; }  /* duration set inline per build */

  .layout { display: grid; grid-template-columns: 15rem minmax(0, 46rem); gap: 1.25rem; max-width: 68rem; margin: 0 auto; padding: 5.4rem .75rem 3.5rem; }
  .sidebar { position: sticky; top: 5rem; align-self: start; max-height: calc(100vh - 7.6rem); overflow-y: auto;
             background: var(--grey); border: 3px outset #fff; padding: .5rem; font: 12px Verdana, Tahoma, sans-serif; }
  .toc-title { font-weight: bold; background: var(--titlebar); color: #fff; padding: .2rem .4rem; margin: -.5rem -.5rem .5rem; }
  .toc li a { padding: .15rem .3rem; color: #000; }
  .toc li.d3 a { padding-left: 1.2rem; }
  .toc li a.active { background: var(--navy); color: #fff; }
  .toc li a:hover { text-decoration: underline; }

  main { background: #fff; border: 3px outset #fff; padding: 1.25rem 1.75rem; box-shadow: 6px 6px 0 #004c4c; }
  main h1 { font: bold 30px "Times New Roman", Times, serif; color: var(--navy); margin: 0 0 1rem; padding-bottom: .35rem;
            border: 0; border-bottom: 5px solid transparent; border-image: linear-gradient(90deg, #ff0000, #ff8800, #ffff00, #00cc00, #0000ff, #8800cc) 1; }
  main h2 { font: bold 21px "Times New Roman", Times, serif; color: var(--navy); margin-top: 2rem; border-bottom: 1px solid var(--dark); }
  main h3 { font: bold 16px "Times New Roman", Times, serif; }
  main h2, main h3 { scroll-margin-top: 5.2rem; }
  main p strong { color: var(--navy); }
  hr { border: 0; border-top: 1px solid var(--dark); border-bottom: 1px solid #fff; }
  blockquote { border-left: 4px solid var(--dark); margin-left: 0; padding-left: 1rem; }

  img { border: 3px inset #fff; }
  .img-placeholder { border: 8px solid; border-image: repeating-linear-gradient(45deg, #ffcc00 0 14px, #000 14px 28px) 8; background: #f0f0f0; color: #000; font: 12px Verdana, Tahoma, sans-serif; }
  .img-placeholder strong { font: bold 14px "Times New Roman", Times, serif; }
  .img-placeholder strong::before { content: "\\1F6A7  "; }
  pre { padding: .75rem; background: #f0f0f0; border: 2px inset #fff; font: 12px/1.45 "Courier New", Courier, monospace; }
  code { font-family: "Courier New", Courier, monospace; font-size: .95em; }
  :not(pre) > code { background: #f0f0f0; padding: 0 .2rem; }
  table { font: 13px Verdana, Tahoma, sans-serif; border: 2px outset #fff; }
  th, td { padding: .3rem .5rem; border: 1px solid var(--dark); }
  th { background: var(--grey); }
  mark { background: #ffff00; border: 2px solid; border-image: repeating-linear-gradient(45deg, #ffcc00 0 6px, #000 6px 12px) 2; padding: 0 .25rem; }
  .placeholder { color: #444; }

  .statusbar { display: flex; position: fixed; bottom: 0; left: 0; right: 0; height: 1.7rem; align-items: center; gap: .4rem; padding: 0 .4rem; z-index: 9;
               background: var(--grey); border-top: 2px solid #fff; font: 11px Verdana, Tahoma, sans-serif; }
  .statusbar .cell { border: 1px inset #fff; padding: .1rem .5rem; }
  .statusbar .grow { flex: 1; }
  .statusbar .cursor { animation: blink 1s steps(1) infinite; }
}
`;

/** Render every page into build/index.html and return the HTML. */
let lastMissing = null;
export function render({ dev = false } = {}) {
  missing.length = 0;
  const rendered = PAGES.map(renderPage);
  const report = missing.join("\n  ");
  if (missing.length && report !== lastMissing) console.warn(`placeholders rendered for missing files:\n  ${report}`);
  lastMissing = report;   // in dev mode, only repeat the warning when the list changes
  const built = new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });

  // The retro ticker: site chrome, then facts. Kept short so each fits on screen while it passes.
  const tickerItems = [
    `Welcome to ${TITLE}`,
    "This page is under construction",
    "1955: the Dartmouth proposal reckoned ten people could make a significant advance in AI over one summer",
    "π has been computed to more than 100 trillion digits; NASA's JPL navigates the solar system with 15 decimals",
    "1958: the New York Times reported the perceptron would soon walk, talk, see, write and be conscious of its existence",
    "The word robot comes from Karel Čapek's 1920 play R.U.R., from the Czech robota, forced labour",
    "1965: Herbert Simon predicted machines would do any work a man can do within twenty years",
    "About 40 digits of π size the observable universe to the width of a hydrogen atom",
    "1966: MIT assigned computer vision to undergraduates as a summer project",
    "Unimate, the first industrial robot, started work on a General Motors line in 1961",
    "1967: Marvin Minsky wrote that artificial intelligence would be substantially solved within a generation",
    "Position 762 of π starts six 9s in a row, the Feynman point",
    "1969: Minsky and Papert showed a single-layer perceptron cannot learn XOR, and neural network research went quiet for a decade",
    "Asimov's Three Laws of Robotics first appeared in the 1942 story Runaround",
    "1970: Minsky told Life magazine a machine with the intelligence of an average human was three to eight years away",
    "In 1897 the Indiana legislature nearly passed a bill that would have made π equal to 3.2",
    "Shakey (1966 to 1972) was the first mobile robot that could reason about its own actions",
    "ELIZA's creator watched his own secretary ask him to leave the room so she could talk to it in private",
    "The first webcam went online in 1993 so people could check a coffee pot at Cambridge",
    "1997: Deep Blue beat Kasparov, and Sojourner became the first rover to drive on Mars",
    "Moravec's paradox: the hard problems are easy and the easy problems are hard",
    "π Day, March 14, is also Einstein's birthday",
  ];
  const tickerText = `★ ${tickerItems.join(" ★ ")} ★`;
  const tickerSeconds = Math.round((tickerText.length * 7.5 + 1400) / 95);   // ~95 px/s regardless of length

  const html = `<!doctype html>
<html lang="en" data-theme="${DEFAULT_THEME}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${TITLE}</title>
${faviconTag}
<script>try { const t = localStorage.getItem("vla-theme"); if (t) document.documentElement.dataset.theme = t; } catch {}</script>
<style>${SHARED_CSS}${MODERN_CSS}${RETRO_CSS}</style>
</head>
<body>
<header class="topbar">
  <div class="brand">${TITLE}</div>
  <nav class="pages">${rendered.map((r) => r.link).join("")}</nav>
  <div class="spacer"></div>
  <button id="theme-toggle" class="theme-toggle" type="button"></button>
  <div class="win-buttons" aria-hidden="true"><span>_</span><span>□</span><span>×</span></div>
</header>
<div class="ticker" aria-hidden="true"><span style="animation-duration: ${tickerSeconds}s">${escapeHtml(tickerText)}</span></div>
<div class="layout">
  <aside class="sidebar">${rendered.map((r) => r.toc).join("\n")}</aside>
  <main>${rendered.map((r) => r.section).join("\n")}</main>
</div>
<script>
(() => {
  // Theme toggle: both stylesheets are in the page; the choice is remembered in this browser.
  const root = document.documentElement;
  const button = document.getElementById("theme-toggle");
  const label = () => { button.textContent = root.dataset.theme === "retro" ? "Switch to modern" : "Switch to 1996"; };
  button.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "retro" ? "modern" : "retro";
    try { localStorage.setItem("vla-theme", root.dataset.theme); } catch {}
    label();
  });
  label();

  // Page routing by hash, and a table of contents that follows the scroll position.
  const pages = [...document.querySelectorAll("section.page")];
  const ids = pages.map((p) => p.dataset.id);
  let onScroll = null;

  const pageFromHash = () => {
    const h = location.hash.slice(1);
    return ids.find((p) => h === p || h.startsWith(p + "-")) || ids[0];
  };

  function watchHeadings(id) {
    if (onScroll) window.removeEventListener("scroll", onScroll);
    const headings = [...document.querySelectorAll('section.page[data-id="' + id + '"] h2, section.page[data-id="' + id + '"] h3')];
    const links = [...document.querySelectorAll('nav.toc[data-page="' + id + '"] a')];
    onScroll = () => {
      let current = headings.length ? headings[0].id : null;
      for (const h of headings) if (h.getBoundingClientRect().top <= 90) current = h.id;
      for (const a of links) a.classList.toggle("active", a.getAttribute("href") === "#" + current);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  function show() {
    const id = pageFromHash();
    pages.forEach((p) => (p.hidden = p.dataset.id !== id));
    document.querySelectorAll("nav.toc").forEach((t) => (t.hidden = t.dataset.page !== id));
    document.querySelectorAll(".topbar a[data-page]").forEach((a) => a.classList.toggle("active", a.dataset.page === id));
    const target = document.getElementById(location.hash.slice(1));
    if (target) target.scrollIntoView({ block: "start" });
    else window.scrollTo(0, 0);
    watchHeadings(id);
  }

  window.addEventListener("hashchange", show);
  show();
})();
</script>
${dev ? LIVE_RELOAD : ""}</body>
</html>
`;

  const build = join(root, "build");
  mkdirSync(build, { recursive: true });
  writeFileSync(join(build, "index.html"), html);
  return html;
}

/** Encrypt build/index.html -> site/index.html. The salt lives in .staticrypt.json; commit it. */
export function encrypt() {
  // On a build server there is no one to answer StatiCrypt's prompt, so fail fast instead of hanging.
  if (!process.env.STATICRYPT_PASSWORD && (process.env.CI || process.env.VERCEL)) {
    throw new Error("STATICRYPT_PASSWORD is not set; add it to the project's environment variables");
  }
  // --short skips StatiCrypt's interactive "password is short, continue?" prompt, which would hang a
  // build server; the README still asks for a 16+ character passphrase.
  execFileSync(
    join(root, "node_modules", ".bin", "staticrypt"),
    ["index.html", "-d", "../site", "--config", "../.staticrypt.json", "--remember", "30", "--short", "--template-title", TITLE],
    { cwd: join(root, "build"), stdio: "inherit" },
  );
  // The passphrase prompt is StatiCrypt's own template; give it the same favicon as the page behind it.
  const out = join(root, "site", "index.html");
  writeFileSync(out, readFileSync(out, "utf8").replace("</title>", `</title>\n        ${faviconTag}`));
}

// `node build.mjs` (npm run build): render, then encrypt. Importers (dev.mjs) call render() themselves.
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  render();
  encrypt();
}
