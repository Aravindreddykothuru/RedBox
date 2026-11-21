// End-to-end walk through Red Box in a headless Chromium browser (Edge or Chrome): sign up -> onboarding
// -> feed -> rate -> movie -> search -> lab -> results -> library. Needs the API and the web app running.
//   npm run e2e        env: BASE (default http://localhost:3000), BROWSER (path to msedge or chrome)
// Screenshots go to e2e/shots/ (git-ignored). Problems are printed at the end and fail the run.
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.BASE ?? "http://localhost:3000";
const BROWSER = process.env.BROWSER ?? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const SHOTS = fileURLToPath(new URL("./shots/", import.meta.url));
mkdirSync(SHOTS, { recursive: true });

// Responses this journey provokes on purpose (a wrong password; the operators-only page for a normal user).
const EXPECTED = [
  { status: 401, path: "/auth/login" },
  { status: 403, path: "/ops/metrics" },
];
const expected = (url, status) =>
  EXPECTED.some((e) => url.includes(e.path) && (status === undefined || status === e.status));

const problems = [];
const browser = await chromium.launch({ executablePath: BROWSER, headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
// the opening animation is checked on its own below; the journey's tabs start as if they had seen it
await context.addInitScript(() => {
  try {
    sessionStorage.setItem("redbox-intro-seen", "1");
  } catch {}
});
const page = await context.newPage();
page.on("console", (m) => {
  // the browser logs every 4xx response as a console error; skip the ones listed above
  if (m.type() === "error" && !expected(m.location().url ?? "")) problems.push(`console: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("requestfailed", (r) => {
  // ERR_ABORTED = the page navigated away while a request was in flight (e.g. after signing out)
  if (r.failure()?.errorText !== "net::ERR_ABORTED") problems.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`);
});
page.on("response", (r) => {
  if (r.status() >= 400 && !expected(r.url(), r.status())) problems.push(`HTTP ${r.status()}: ${r.url()}`);
});

const shot = (name, full = false) => page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage: full });
const step = (msg) => console.log(`- ${msg}`);
// Posters load lazily as a viewer scrolls. Before a full-page shot, scroll through the page and wait until
// the posters in view (the rows' visible width) have loaded or failed; returns how many loaded.
const settlePosters = async () => {
  await page.evaluate(async () => {
    for (let y = 0; y < document.documentElement.scrollHeight; y += 500) {
      window.scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 100));
    }
    window.scrollTo(0, 0);
  });
  const inView = () =>
    [...document.querySelectorAll("main img")].filter((i) => {
      const r = i.getBoundingClientRect();
      return r.width > 0 && r.left < window.innerWidth && r.right > 0;
    });
  await page
    .waitForFunction(`(${inView})().every((i) => i.complete)`, null, { timeout: 20000, polling: 250 })
    .catch(() => {});
  await page.waitForTimeout(400); // the fade-in
  return page.evaluate(`(() => { const all = (${inView})(); return { total: all.length, loaded: all.filter((i) => i.naturalWidth > 0).length }; })()`);
};
const titlesOf = async (section) => page.locator(`section[aria-label="${section}"] p[data-film-title]`).allTextContents();

try {
  // The opening animation: plays in a new tab, lifts by itself, does not play again after a reload, and
  // any key skips it. Frames go to shots/00*.
  {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const p = await ctx.newPage();
    p.on("pageerror", (e) => problems.push(`pageerror (intro): ${e.message}`));
    await p.goto(BASE, { waitUntil: "commit" }); // the animation starts at the first paint, not at "load"
    const intro = p.getByTestId("intro");
    await intro.waitFor({ state: "visible" });
    const t0 = Date.now();
    // strands, the R, the R shining, the name unfolding, the name, the rush toward the viewer
    const frames = [[250, "a"], [1100, "b"], [1700, "c"], [2350, "d"], [2900, "e"], [3550, "f"]];
    for (const [at, name] of frames) {
      await p.waitForTimeout(Math.max(0, at - (Date.now() - t0)));
      await p.screenshot({ path: join(SHOTS, `00${name}-intro.png`) });
    }
    await intro.waitFor({ state: "detached", timeout: 5000 });
    await p.getByText("Rate ten films, get picks").waitFor();
    await p.waitForTimeout(500);
    await p.screenshot({ path: join(SHOTS, "00g-after-intro.png") });
    await p.reload();
    await p.getByText("Rate ten films, get picks").waitFor();
    if (await intro.isVisible()) problems.push("the intro played again after a reload in the same tab");

    const p2 = await ctx.newPage(); // a new tab plays it again
    await p2.goto(BASE, { waitUntil: "commit" });
    await p2.getByTestId("intro").waitFor({ state: "visible" });
    await p2.waitForTimeout(500);
    const skipAt = Date.now();
    await p2.keyboard.press("Escape");
    await p2.getByTestId("intro").waitFor({ state: "detached", timeout: 1500 });
    const skipMs = Date.now() - skipAt;
    // after a skip the page must be there at once, not still waiting for the intro's delayed entrance
    const heroOpacity = await p2.locator(".enter").first().evaluate((el) => getComputedStyle(el).opacity);
    if (heroOpacity !== "1") problems.push(`after skipping the intro the hero is not visible (opacity ${heroOpacity})`);
    step(`opening animation plays, lifts by itself, skips in ${skipMs} ms`);
    await ctx.close();
  }

  await page.goto(BASE);
  await page.getByText("Rate ten films, get picks").waitFor();
  await page.getByRole("heading", { name: "Trending now" }).waitFor();
  await page.waitForTimeout(800);
  await shot("01-landing");
  await page.getByRole("button", { name: "What is Red Box?" }).click();
  await page.getByText("It does not stream films").waitFor();
  await settlePosters();
  await shot("01b-landing-full", true);
  step("landing, top ten and FAQ ok");

  // a real account (API in "local" auth mode); a fresh email per run, entered on the landing page
  const email = `e2e-${Date.now()}@example.com`;
  const password = "a long test password";
  await page.locator("#hero-email").fill(email);
  await page.getByRole("button", { name: "Get started" }).first().click();
  await page.getByRole("heading", { name: "Create your account" }).waitFor();
  if ((await page.getByLabel("Email", { exact: true }).inputValue()) !== email) problems.push("email not carried over from the landing page");
  await shot("01c-signup");
  await page.getByLabel("Name", { exact: true }).fill("Asha");
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Create account" }).click();
  await page.getByRole("heading", { name: /what do you like to watch/i }).waitFor();
  step(`signed up as ${email}`);
  for (const g of ["Sci-Fi", "Thriller", "Comedy"]) await page.getByRole("button", { name: g, exact: true }).click();
  await shot("02-onboarding-genres");
  await page.getByRole("button", { name: "Next: rate some films" }).click();
  const sliders = page.locator('[role="slider"]');
  await sliders.first().waitFor();
  await page.waitForTimeout(500);
  const stars = [5, 4, 5, 2, 5, 4, 1, 5, 4, 5, 3, 5];
  for (let i = 0; i < stars.length; i++) {
    await sliders.nth(i).locator("button").nth(stars[i] - 1).click({ position: { x: 16, y: 10 } });
  }
  await shot("03-onboarding-rated");
  step(`rated ${stars.length} seed films`);
  await page.getByRole("button", { name: "Show my picks" }).click();
  await page.waitForURL("**/home");
  await page.getByText("Top picks for you").waitFor({ timeout: 20000 });
  const posters = await settlePosters();
  if (posters.loaded < 0.9 * posters.total) problems.push(`only ${posters.loaded} of ${posters.total} posters loaded on home`);
  step(`home posters: ${posters.loaded} of ${posters.total} loaded (the rest show their cover)`);
  await shot("04-home", true);
  const before = await titlesOf("Top picks for you");
  step(`top picks: ${before.slice(0, 5).join(" | ")}`);

  // real-time update: rate the first top pick, the list must change and drop it
  const firstRow = page.locator('section[aria-label="Top picks for you"]');
  await firstRow.locator('[role="slider"]').first().locator("button").nth(4).click({ position: { x: 10, y: 8 } });
  await page.waitForTimeout(2500);
  const after = await titlesOf("Top picks for you");
  step(`after rating "${before[0]}" 5 stars: ${after.slice(0, 5).join(" | ")}`);
  if (after.includes(before[0])) problems.push("rated film still in top picks");
  if (JSON.stringify(after) === JSON.stringify(before)) problems.push("top picks did not change after a rating");

  // diversity slider
  const range = page.locator('input[type="range"]');
  await range.focus();
  for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(2000);
  const diverse = await titlesOf("Top picks for you");
  step(`diversity 0.4: ${diverse.slice(0, 5).join(" | ")}`);
  await settlePosters();
  await shot("05-home-diverse");

  // movie page
  await firstRow.locator("a").first().click();
  await page.waitForURL("**/movie/**");
  await page.getByText("Why it's here").waitFor();
  // "Why this pick?": the ranker's score split by signal (TreeSHAP)
  await page.getByRole("heading", { name: "How the ranker scored it for you" }).waitFor({ timeout: 15000 });
  const signals = await page.locator('section[aria-labelledby="why-title"] li span.truncate').allTextContents();
  if (!signals.length) problems.push("the Why-this-pick breakdown shows no signals");
  await page.waitForTimeout(1200);
  await settlePosters();
  await shot("06-movie", true);
  await page.getByRole("radio", { name: "Similar story and themes" }).click();
  await page.getByText("More like this").waitFor();
  await page.waitForTimeout(800);
  await settlePosters();
  await shot("07-movie-content-similar", true);
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/watchlist/") && r.request().method() === "PUT"),
    page.getByRole("button", { name: /Add to watchlist/ }).click(),
  ]);
  await page.getByText("On your watchlist").waitFor();
  step("movie page, similar tabs and watchlist ok");

  // search
  await page.goto(`${BASE}/search`);
  await page.getByRole("button", { name: "time travel" }).click();
  await page.waitForTimeout(2500);
  await shot("08-search-describe");
  const hits = await page.locator("main p[data-film-title]").allTextContents();
  step(`semantic "time travel": ${hits.slice(0, 5).join(" | ")}`);
  await page.getByRole("radio", { name: "By title" }).click();
  await page.getByLabel("Movie title").fill("matrix");
  await page.waitForTimeout(1500);
  const titleHits = await page.locator("main p[data-film-title]").allTextContents();
  step(`title "matrix": ${titleHits.join(" | ")}`);

  // lab
  await page.goto(`${BASE}/lab`);
  await page.getByText("How much do the models agree?").waitFor({ timeout: 20000 });
  await page.waitForTimeout(800);
  await shot("09-lab", true);
  // MovieLens viewers are offered only where the bundle has their histories: a public deployment leaves
  // them out (the MovieLens licence forbids redistributing them), so there the Lab compares on "you" only
  const viewers = await page.getByLabel("Viewer").locator("option").count();
  if (viewers > 3) {
    await page.getByLabel("Viewer").selectOption({ index: 3 });
    await page.waitForTimeout(2000);
    await shot("10-lab-viewer");
    step("lab ok");
  } else {
    step("lab ok (signed-in viewer only: this deployment has no MovieLens histories)");
  }

  // results
  await page.goto(`${BASE}/results`);
  await page.getByText("Ranking accuracy").waitFor();
  await page.waitForTimeout(1500);
  await shot("11-results", true);
  await page.getByRole("radio", { name: "Amazon Video Games" }).click();
  await page.waitForTimeout(1200);
  await shot("12-results-amazon");
  await page.getByRole("radio", { name: "MovieLens-32M (live app)" }).click();
  await page.getByText("p < 10⁻⁸⁶").waitFor();
  await page.waitForTimeout(1200);
  await shot("12b-results-ml32m");
  step("results ok (with the live app's ML-32M tab and its significance)");

  // library
  await page.goto(`${BASE}/library`);
  await page.getByText("counted as liked").waitFor();
  await shot("13-library");
  await page.getByRole("radio", { name: "Watchlist" }).click();
  await page.waitForTimeout(800);
  await shot("14-watchlist");
  step("library ok");

  // watch-history import: a small Netflix file with day/month dates, Hindi and Telugu tags, a renamed
  // title, a TV episode and a title the catalogue does not have
  await page.goto(`${BASE}/import`);
  await page.getByRole("heading", { name: "Import your watch history" }).waitFor();
  await shot("14c-import");
  const netflixCsv = [
    '"Title","Date"',
    '"RRR (Hindi)","20/11/2026"',
    '"Stranger Things: Season 4: Chapter One: The Hellfire Club","19/11/2026"',
    '"Baahubali 2: The Conclusion (Telugu)","18/11/2026"',
    '"Dangal","12/11/2026"',
    '"Spider-Man: Into the Spider Verse","02/11/2026"',
    '"A Film That Does Not Exist","01/11/2026"',
  ].join("\n");
  await page
    .locator('input[type="file"]')
    .setInputFiles({ name: "NetflixViewingHistory.csv", mimeType: "text/csv", buffer: Buffer.from(netflixCsv) });
  await page.getByRole("heading", { name: "4 films found in your Netflix history" }).waitFor();
  await page.getByText("1 TV episode left out").waitFor();
  await page.getByText("1 title not in the catalogue").waitFor();
  await shot("14d-import-preview", true);
  await page.getByRole("button", { name: /^Add \d+ films?$/ }).click();
  await page.getByRole("heading", { name: "Your history is in" }).waitFor();
  await page.goto(`${BASE}/home`);
  await page.getByText(/imported films?\. Rate a film/).waitFor({ timeout: 20000 });
  await page.goto(`${BASE}/library?tab=imported`);
  await page.getByText(/imported · they count as liked/).waitFor();
  const imported = await page.locator("main ul li a.line-clamp-1").allTextContents();
  await shot("14e-import-library");
  step(`watch-history import: ${imported.join(" | ")}`);

  // watch together: you and a friend who loves Inception
  await page.goto(`${BASE}/group`);
  await page.getByRole("heading", { name: "Watch together" }).waitFor();
  await page.getByPlaceholder("Friend 1").fill("Ravi");
  await page.getByRole("textbox", { name: "Add a film Ravi loves" }).fill("Inception");
  await page.getByRole("button", { name: /^Inception/ }).filter({ visible: true }).first().click();
  await page.getByRole("radio", { name: "Nobody unhappy" }).click();
  await page.getByRole("button", { name: "Find films for us" }).click();
  await page.getByRole("heading", { name: "Films for You and Ravi" }).waitFor({ timeout: 20000 });
  const together = await page.locator("main ol li a.line-clamp-1").allTextContents();
  if (together.length < 5) problems.push(`watch together returned ${together.length} films`);
  await settlePosters();
  await shot("14f-watch-together", true);
  step(`watch together (least misery): ${together.slice(0, 5).join(" | ")}`);

  // account: sign out from the account menu, wrong password refused, sign back in, account page
  await page.getByRole("button", { name: "Account menu" }).click();
  await shot("14b-account-menu");
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.waitForURL(BASE + "/");
  await page.goto(`${BASE}/login`);
  await page.getByLabel("Email", { exact: true }).fill(email);
  await page.getByLabel("Password", { exact: true }).fill("not the password");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByText("wrong email or password", { exact: false }).waitFor();
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL("**/home");
  await page.goto(`${BASE}/account`);
  await page.getByText(`Signed in as ${email}`).waitFor();
  await shot("15-account");
  step("sign out, wrong password refused, sign in again, account page ok");

  // operators' live metrics (the test account is an operator when the API lists it, otherwise 403 page)
  await page.goto(`${BASE}/ops`);
  await page.getByText(/Live metrics|Operators only/).first().waitFor();
  await shot("16-ops", true);
  step("operations page ok");

  // the interface in Telugu, then back to English (the choice is remembered in the browser)
  await page.goto(`${BASE}/home`);
  await page.getByText("Top picks for you").waitFor({ timeout: 20000 });
  // the row loads on its own, after the top picks: wait for it (counting at once raced it and skipped every
  // language check below). A bundle without new films sets NO_NEW_FILMS=1.
  const newFilms = await page
    .getByRole("heading", { name: /^New (.+ )?releases for you$/ })
    .first()
    .waitFor({ timeout: 20000 })
    .then(() => true)
    .catch(() => false);
  if (!newFilms && !process.env.NO_NEW_FILMS) problems.push("no new-releases row on home");
  step(newFilms ? "new-releases row shown on home" : "no new-releases row");
  // the new films' language filter: each language's own films, with reasons that name it
  const languages = page.getByRole("radiogroup", { name: "Language" });
  if (newFilms && (await languages.count())) {
    for (const lang of ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam"]) {
      await languages.getByRole("radio", { name: lang, exact: true }).click();
      const row = page.locator(`div[aria-busy="false"] > section[aria-label="New ${lang} releases for you"]`);
      await row.waitFor({ timeout: 20000 });
      const titles = await row.locator("p[data-film-title]").allTextContents();
      const reasons = await row.getByText(/^New in /).allTextContents();
      if (titles.length < 10) problems.push(`only ${titles.length} new ${lang} films`);
      if (reasons.some((r) => !r.startsWith(`New in ${lang}:`))) problems.push(`a new ${lang} film names another language`);
      await row.scrollIntoViewIfNeeded();
      await page
        .waitForFunction(
          (label) =>
            [...document.querySelectorAll(`section[aria-label="${label}"] img`)]
              .filter((i) => i.getBoundingClientRect().left < window.innerWidth)
              .every((i) => i.complete),
          `New ${lang} releases for you`,
          { timeout: 15000 },
        )
        .catch(() => {});
      await page.waitForTimeout(400); // the posters' fade-in
      await row.screenshot({ path: join(SHOTS, `16a-new-films-${lang.toLowerCase()}.png`) });
      if (lang === "English") {
        // the report's figure: the row with its filter, below the fixed header
        await row.evaluate((el) => {
          el.scrollIntoView({ block: "start" });
          window.scrollBy(0, -96);
        });
        await page.waitForTimeout(300);
        await shot("16a-home-english");
      }
      step(`new ${lang} films: ${titles.slice(0, 4).join(" | ")}`);
    }
    await languages.getByRole("radio", { name: "All", exact: true }).click();
    await page.getByRole("heading", { name: "New releases for you" }).waitFor();
  }
  await page.getByRole("button", { name: "Language: English" }).click();
  await page.getByRole("button", { name: "తెలుగు" }).click();
  await page.getByText("మీ కోసం అగ్ర సూచనలు").waitFor();
  if ((await page.evaluate(() => document.documentElement.lang)) !== "te") problems.push("<html lang> not set to te");
  await page.waitForTimeout(800);
  await settlePosters();
  await shot("16b-home-telugu");
  await page.getByRole("button", { name: "భాష: తెలుగు" }).click();
  await page.getByRole("button", { name: "English" }).click();
  await page.getByText("Top picks for you").waitFor();
  step("language switch: Telugu and back to English");

  // phone width
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(`${BASE}/home`);
  await page.getByText("Top picks for you").waitFor({ timeout: 20000 });
  await page.waitForTimeout(1200);
  await shot("17-mobile-home");
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  if (overflow) problems.push("horizontal page overflow at 390px");
  await page.goto(`${BASE}/results`);
  await page.getByText("Ranking accuracy").waitFor();
  await page.waitForTimeout(1200);
  await shot("18-mobile-results", true);
  const overflow2 = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  if (overflow2) problems.push("horizontal page overflow on /results at 390px");
} catch (e) {
  problems.push(`FAILED: ${e.message.split("\n").slice(0, 12).join("\n  ")}`);
  await shot("zz-failure", true).catch(() => {});
} finally {
  await browser.close();
}
console.log(problems.length ? `\nPROBLEMS (${problems.length}):\n${problems.join("\n")}` : "\nno problems");
process.exitCode = problems.length ? 1 : 0;
