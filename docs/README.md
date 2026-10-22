# Docs

The writeup is one public static page with a navbar and a per-page table of contents, at
<https://axibo-vl-atest.vercel.app/>. Each navbar entry is a Markdown file in `pages/` (`task1.md` …
`task4.md`, listed in `build.mjs`); figures go in `images/`; example session files shown in the writeup
go in `data/`. On every push, Vercel runs `npm run build` in this folder, which renders the pages into
`site/index.html`, copies `images/` beside it and writes the favicon, `robots.txt` and `sitemap.xml`.
Only `site/` is served.

## Writing

- `## ` and `### ` headings become the table of contents, which highlights the section in view as you scroll.
- Files worth showing are linked to GitHub rather than embedded. Session files referenced from the writeup are expected under `data/` in the repo (copy them from a recorded session; the originals under `datasets/` are gitignored).
- Images: `![caption](images/name.png)`; they are served as files next to the page and lazy-loaded, so large GIFs are fine.
- `<mark>text</mark>` highlights a placeholder still to fill in.

## Title, description and search engines

The site's name, tagline, description, keywords, organization, the masthead text and the links under it
are the constants at the top of `build.mjs`. They feed the `<title>`, the meta description and keywords,
the Open Graph and Twitter tags for link previews, a JSON-LD `TechArticle` block, the masthead at the top
of the page, and `robots.txt` and `sitemap.xml`. `SITE_URL` must be the deployed address, since the
canonical link and the preview image are absolute. The document title changes with the task as you
navigate (`Task 2: Fine-tuning OpenVLA | Axibo Robotics VLA Test`).

The link-preview image is `images/og.png`, the first frame of `images/engine.gif`. To regenerate on macOS:
`sips -s format png images/engine.gif --out images/og.png`

The favicon is `images/favicon.png`, a 64px square crop of `images/GreatApe.jpg`. To regenerate from a new
image on macOS: `sips -s format png -c 350 350 --cropOffset 80 80 GreatApe.jpg --out favicon.png && sips -Z 64 favicon.png`

## One-time setup (Vercel)

1. Vercel dashboard > Add New > Project > import the GitHub repo.
2. Set **Root Directory** to `docs`. The framework, build command (`npm run build`) and output directory (`site`) come from `vercel.json`, so leave them as detected.
3. Deploy. Every later push to the production branch redeploys; other branches get preview URLs.

No environment variables are needed. (The site used to be passphrase-protected with StatiCrypt; a leftover
`STATICRYPT_PASSWORD` variable in the Vercel project can be deleted.)

## Write with a live preview

```
cd docs
npm install
npm run dev        # then open http://127.0.0.1:8000
```

The dev server re-renders on every save to `pages/`, `images/`, `data/` or `../source/config.py` and the
browser reloads itself. Changes to `build.mjs` itself (theme, layout, metadata) need the dev server restarted.

## Build the site

```
npm run build
python3 -m http.server --directory site --bind 127.0.0.1 8000   # optional: check the output
```
