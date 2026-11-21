# Red Box web app

The Red Box website, built with Next.js 16 (App Router), TypeScript, Tailwind CSS 4, TanStack Query and
Recharts. Every page fetches its data from the API ([../backend](../backend)) in the browser.

## Run locally

```bash
# 1. the API, in another terminal
cd backend && uv run uvicorn app.main:app

# 2. the website
cd frontend
cp .env.example .env.local     # defaults point at the local API, in demo-auth mode
npm install
npm run dev                    # http://localhost:3000
```

| Command | What it does |
|---|---|
| `npm run dev` | Development server with hot reload |
| `npm run build && npm start` | Production build and server |
| `npm run lint` · `npm run typecheck` | ESLint and TypeScript checks (also run in CI) |
| `npm run sync-results` | Copy the offline result tables and report figures from `docs/` into `public/` |
| `npm run e2e` | Browser journey (sign-up → onboarding → rate → search → lab → results) in headless Edge/Chrome against the running stack; set `BROWSER` to the browser's path if needed |

## Pages

| Route | Page |
|---|---|
| `/` | Landing page: email sign-up, trending top ten, reasons to join, FAQ (with live test results from the API) |
| `/login` | Sign in or create an account (the email typed on the landing page carries over); a name-only profile in demo mode |
| `/onboarding` | Pick genres, then rate seed films (or import a watch history instead) |
| `/import` | Import a watch history from Netflix, IMDb or Letterboxd: instructions per service, a preview of the matched films to tick or untick, then save |
| `/home` | Billboard of your top pick (over its own poster, blurred), then rows: top picks (hybrid, with a diversity slider), new Indian films (a filter for Telugu, Tamil, Kannada, Malayalam), "Because you liked…", hidden gems, a top-ten row. A brand-new user gets "Picked for you to start" (the cold-start forecast). Every card can be rated in place |
| `/movie/[id]` | Poster, Wikipedia summary, viewer keywords, your rating, watchlist, why it was recommended, "How the ranker scored it for you" (TreeSHAP), similar films (collaborative or content) |
| `/group` | Watch together: you plus up to five friends (each described by a few films they love), a group strategy (average, least misery, most pleasure), and every pick's fit for each person |
| `/search` | Natural-language search ("describe it") or title search with a genre filter |
| `/library` | Your ratings (editable), watchlist and imported history (`?tab=imported`) |
| `/lab` | Model Lab: every model's top 10 for the same viewer, the overlap heatmap, the MMR diversity slider |
| `/results` | Results dashboard: offline experiments on MovieLens-1M and Amazon, and the live app's MovieLens-32M models (with significance), as charts and a table |

## Sign-in modes (`NEXT_PUBLIC_AUTH_MODE`)

- **local (default):** email + password accounts handled by the API (`REDBOX_AUTH_MODE=local`).
- **supabase:** set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`; the API verifies Supabase
  tokens (`REDBOX_AUTH_MODE=supabase`).
- **demo:** a name kept in the browser, sent as `X-User-Id`; the API must run with `REDBOX_AUTH_MODE=dev`.

## Design

A streaming-service layout under the Red Box name and colour. It borrows the familiar patterns, not
another brand's logo, typeface, exact red, copy or artwork.

- **Name:** "Red Box" lives in one constant, `src/lib/brand.ts`, so the visible name is a one-line
  change. The internal names follow it: storage keys, `REDBOX_*` settings, package names, and the hybrid's
  name in the result tables, "Red Box hybrid".

- **Opening animation** (`components/intro.tsx`, CSS in `globals.css`, about 4 seconds):
  - Red light strands sweep in and fuse into a big R, and a shine crosses it.
  - The R then shrinks into place as the rest of the name, RED BOX, unfolds out of it.
  - The name rushes toward the viewer as the site appears.
  - It plays once per browser tab, on whichever page is opened first, and any key, click, tap or scroll
    skips it.
  - It is skipped for "reduce motion" and is not shown without JavaScript. A head script
    (`lib/intro.ts`) decides before the first paint, so the page never flashes first.
  - It has no sound, because browsers block sound until the visitor interacts.
- **Languages** (`src/lib/i18n.tsx`): English, Telugu and Hindi, chosen from the header and remembered in the
  browser.
  - The interface, genre names and the models' short reasons under each film are translated, and Noto
    Sans Telugu and Devanagari are loaded only when needed.
  - Film titles and keywords, and the technical pages (Model Lab, Results, Operations), stay in English.
- **Installable app (PWA):**
  - `src/app/manifest.ts`, icons in `public/icons/` and `public/sw.js`.
  - Pages always come from the network, with `offline.html` when there is none; build files are cached.
  - "Install the app" appears in the account menu when the browser offers it.
- **Posters:** from Wikipedia (`recsys wiki`), for 97% of the catalogue. Films without one keep the
  typographic cover, and list rows use a one-letter thumbnail (`Poster sizes="thumb"`).
- **Landing:**
  - A full-screen hero with the headline, the offer and an email box ("Get started" carries the address
    to the sign-up form), closed by a curved red line.
  - Then a numbered "Trending now" top ten (the models' forecast of what viewers watch next, live from the API), four "More reasons to join" cards, an FAQ
    accordion, a second email box and a link footer.
- **App shell:**
  - The header is fixed and see-through over the top of each page, turning solid on scroll.
  - Signed in, it shows the links, a search icon and an account menu (Account, My list, Operations for
    operators, Sign out). On narrow screens the links fold into a "Browse" menu.
- **Home:**
  - A billboard for the top pick: title, why it is here, "More info" and "My list".
  - Below it, rows of posters that run edge to edge, with scroll arrows on hover, and a top-ten row with
    big outlined numbers.
  - Posters grow a little on hover.
- **Film page:** a dark full-width header with the poster, details, "Why it's here", rating and watchlist,
  then "Cast:" and "Genres:" lines and the similar-films rows.
- **Colour:**
  - Near-black neutrals (`#141414` page, `#2d2d2d` panels) with white and grey text.
  - A deep cinema red (`#d6283b`, white text on it at 5:1) for the wordmark, the main call to action,
    ratings and accents. It is not Netflix's exact red.
  - Small red text uses a lighter red (`#ff6370`) so it reads on the dark page, and errors are orange.
  - The tab icon is a red R on black (`src/app/icon.svg`).
  - Tokens are in `src/app/globals.css`, with small corners set globally (4–8px).
- **Type:** Geist only, with bold and extra-bold headings and regular body text.
- **Signature:** every film carries one short line saying why it is there ("Because you liked Heat"). In
  rows where that would repeat, it shows a per-film fact instead (similarity).
- **Covers:** without TMDB artwork, every film gets a bold typographic cover tinted by its first genre.
- **Stills wall:** behind the landing hero and sign-in, rows of film stills (`public/hero/stills/`, sizes in
  `src/lib/stills.ts`) are tilted a little and drift slowly, alternate rows the other way, under a dark overlay.
  - Each image is shown whole at its own shape, never cropped.
  - The stills were supplied by the project author for the local demo, and the rights belong to their owners.
    Replace them with licensed artwork, for example TMDB posters with attribution, before any public deployment.
- **Scene image:** `public/hero/cinema.webp`, an empty cinema with a lit screen, sits behind the home
  billboard and the film header. It was AI-generated with Google Gemini, then cropped and converted to WebP.
- **Charts:** one series per chart, with the hybrid highlighted. The colours pass a colour-vision-deficiency
  check against the dark surface, and every chart has a table view.
- **Motion:** one short entrance per page, posters that grow on hover, and a fading header. Everything
  respects `prefers-reduced-motion`.
- **History:** an earlier version was designed with an unslop pass (unslop by Damir Sokolovsky,
  @heydamir, CC BY 4.0).
