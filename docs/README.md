# Docs

The writeup is `index.md`; figures go in `images/`. On every push, Vercel runs `npm run build` in this folder, which renders the Markdown to HTML, inlines the images, and encrypts the page with StatiCrypt into `site/`. Only `site/` is served, behind a passphrase prompt. The Markdown and images are never public.

## One-time setup (Vercel)

1. Vercel dashboard > Add New > Project > import the private GitHub repo.
2. Set **Root Directory** to `docs`. The framework, build command (`npm run build`) and output directory (`site`) come from `vercel.json`, so leave them as detected.
3. Under Environment Variables add `STATICRYPT_PASSWORD` for Production and Preview. Use 16+ characters: the ciphertext is public, so a short passphrase can be brute-forced offline.
4. Deploy. Every later push to the production branch redeploys; other branches get preview URLs with the same passphrase.

Commit `.staticrypt.json`: it holds the salt, and changing it invalidates every visitor's "remember me".

The favicon is `images/favicon.png`, a 64px square crop of `images/GreatApe.jpg`, embedded into both the passphrase prompt and the page. To regenerate from a new image on macOS:
`sips -s format png -c 350 350 --cropOffset 80 80 GreatApe.jpg --out favicon.png && sips -Z 64 favicon.png`

## Preview locally

```
cd docs
npm install
echo 'STATICRYPT_PASSWORD=your-long-passphrase' > .env   # gitignored; omit it to be prompted instead
npm run build
python3 -m http.server --directory site --bind 127.0.0.1 8000   # then open http://127.0.0.1:8000
```
