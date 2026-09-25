# Docs

The writeup is one static page with a navbar and a per-page table of contents. Each navbar entry is a Markdown file in `pages/` (`task1.md` … `task4.md`, listed in `build.mjs`); figures go in `images/`; example session files shown in the writeup go in `data/`. On every push, Vercel runs `npm run build` in this folder, which renders the pages to a single HTML file, inlines the images, and encrypts it with StatiCrypt into `site/`. Only `site/` is served, behind a passphrase prompt. The Markdown, images and data are never public.

## Writing

- `## ` and `### ` headings become the table of contents, which highlights the section in view as you scroll.
- Files worth showing are linked to GitHub rather than embedded. Session files referenced from the writeup are expected under `data/` in the repo (copy them from a recorded session; the originals under `datasets/` are gitignored).
- Images: `![caption](images/name.png)`; they are inlined into the page.
- `<mark>text</mark>` highlights a placeholder still to fill in.

## One-time setup (Vercel)

1. Vercel dashboard > Add New > Project > import the private GitHub repo.
2. Set **Root Directory** to `docs`. The framework, build command (`npm run build`) and output directory (`site`) come from `vercel.json`, so leave them as detected.
3. Under Environment Variables add `STATICRYPT_PASSWORD` for Production and Preview. Use 16+ characters: the ciphertext is public, so a short passphrase can be brute-forced offline.
4. Deploy. Every later push to the production branch redeploys; other branches get preview URLs with the same passphrase.

Commit `.staticrypt.json`: it holds the salt, and changing it invalidates every visitor's "remember me".

The favicon is `images/favicon.png`, a 64px square crop of `images/GreatApe.jpg`, embedded into both the passphrase prompt and the page. To regenerate from a new image on macOS:
`sips -s format png -c 350 350 --cropOffset 80 80 GreatApe.jpg --out favicon.png && sips -Z 64 favicon.png`

## Write with a live preview

```
cd docs
npm install
npm run dev        # then open http://127.0.0.1:8000
```

The dev server re-renders on every save to `pages/`, `images/`, `data/` or `../source/config.py` and the browser reloads itself. It serves the page unencrypted on localhost only, so there is no passphrase while writing. Changes to `build.mjs` itself (theme, layout) need the dev server restarted.

## Build the encrypted site

```
echo 'STATICRYPT_PASSWORD=your-long-passphrase' > .env   # gitignored; omit it to be prompted instead
npm run build
python3 -m http.server --directory site --bind 127.0.0.1 8000   # optional: check the passphrase prompt
```

`site/index.html` is bound to the passphrase at build time: change `.env`, rebuild. Serving `site/` with a static server does not pick up Markdown edits; that is what `npm run dev` is for.
