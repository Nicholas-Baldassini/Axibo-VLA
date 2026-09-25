# Docs

The writeup is `index.md`; figures go in `images/`. On every push to `main` that touches `docs/`, the GitHub Action in `.github/workflows/docs.yml` renders the Markdown to HTML, inlines the images, encrypts the page with StatiCrypt, and uploads it to Cloudflare Pages. The Markdown and images are never served; visitors see a passphrase prompt.

## One-time setup

1. Create the Cloudflare Pages project (direct upload, not git-connected):
   `npx wrangler login && npx wrangler pages project create vla-test --production-branch main`
2. Create a Cloudflare API token with the **Cloudflare Pages: Edit** account permission, and note your account ID from the dashboard sidebar.
3. Add three repo secrets under Settings > Secrets and variables > Actions:
   `STATICRYPT_PASSWORD` (16+ characters; the ciphertext is public, so short ones can be brute-forced offline), `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`.
4. Push. The site is at `https://vla-test.pages.dev`.

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
