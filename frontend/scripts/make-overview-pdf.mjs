// Print docs/overview/RedBox_Technical_Overview.html to docs/RedBox_Technical_Overview.pdf (A4, page numbers),
// with the same headless Edge/Chrome the browser journey uses.
//   node scripts/make-overview-pdf.mjs        env: BROWSER (path to msedge or chrome)
import { chromium } from "playwright-core";
import { fileURLToPath, pathToFileURL } from "node:url";

const BROWSER = process.env.BROWSER ?? "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const source = fileURLToPath(new URL("../../docs/overview/RedBox_Technical_Overview.html", import.meta.url));
const target = fileURLToPath(new URL("../../docs/RedBox_Technical_Overview.pdf", import.meta.url));

const browser = await chromium.launch({ executablePath: BROWSER, headless: true });
const page = await browser.newPage();
await page.goto(pathToFileURL(source).href, { waitUntil: "load" });
await page.pdf({
  path: target,
  format: "A4",
  printBackground: true,
  preferCSSPageSize: true,
  displayHeaderFooter: true,
  headerTemplate: "<span></span>",
  footerTemplate:
    '<div style="width:100%;font-size:7.5px;color:#888;padding:0 17mm;display:flex;justify-content:space-between;font-family:Segoe UI,Arial">' +
    '<span>Red Box: how it works</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>',
});
await browser.close();
console.log(`wrote ${target}`);
