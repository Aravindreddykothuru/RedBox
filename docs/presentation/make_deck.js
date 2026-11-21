// Builds RedBox_final_review.pptx (the final-review presentation) with pptxgenjs.
//   NODE_PATH=<folder with node_modules containing pptxgenjs> node docs/presentation/make_deck.js
// Every number on the slides comes from docs/report/11_results.md and docs/report/10_testing.md.
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const FIG = path.join(HERE, "..", "figures", "app");
const OUT = path.join(HERE, "RedBox_final_review.pptx");
const APPLY_THEME = process.env.APPLY_THEME; // path to the pptx skill's apply_theme.js (optional)

// Dark "screening room" theme, as in the web app. lt1/dk1 are deliberately swapped so that PowerPoint's
// default text (tx1) is light ink and its default background (bg1) is the dark tone.
const THEME = {
  name: "Red Box",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "F3EDE4", // text1: primary ink
    lt1: "16130F", // background1: near-black
    dk2: "CFC5B8", // text2: secondary ink
    lt2: "221D18", // background2: card surface
    accent1: "D6283B", // Red Box red
    accent2: "3987E5", // chart series blue
    accent3: "E8404F", // chart highlight red
    accent4: "9A9083", // muted
    accent5: "3A332B", // hairline
    accent6: "F4EFE6", // paper
    hlink: "FF6370",
    folHlink: "E8404F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "[Your name]";
pres.title = "Red Box: Collaborative Filtering for Personalized Recommendations";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const S = pres.shapes;

// ------------------------------------------------------------------ layouts
const MX = 0.6; // side margin
const CW = 13.333 - 2 * MX; // content width

pres.defineSlideMaster({
  title: "CF_CONTENT",
  background: { color: C.background1 },
  objects: [
    {
      placeholder: {
        options: { name: "eyebrow", type: "body", x: MX, y: 0.45, w: CW, h: 0.35, fontSize: 13, bold: true, color: C.accent1, margin: 0, valign: "top" },
        text: "",
      },
    },
    {
      placeholder: {
        options: { name: "title", type: "title", x: MX, y: 0.8, w: CW, h: 0.95, fontSize: 30, bold: true, color: C.text1, margin: 0, valign: "top", align: "left", fontFace: "Cambria" },
        text: "",
      },
    },
  ],
  slideNumber: { x: 12.23, y: 6.95, w: 0.5, h: 0.3, fontSize: 11, color: HEX.accent4, align: "right" },
});

pres.defineSlideMaster({
  title: "CF_TITLE",
  background: { color: C.background1 },
  objects: [
    {
      placeholder: {
        options: { name: "eyebrow", type: "body", x: MX, y: 1.2, w: CW, h: 0.4, fontSize: 14, bold: true, color: C.accent1, margin: 0 },
        text: "",
      },
    },
    {
      placeholder: {
        options: { name: "title", type: "title", x: MX, y: 1.7, w: CW, h: 1.5, fontSize: 66, bold: true, color: C.text1, margin: 0, valign: "top", align: "left", fontFace: "Cambria" },
        text: "",
      },
    },
    {
      placeholder: {
        options: { name: "subtitle", type: "body", x: MX, y: 3.35, w: 9.6, h: 1.1, fontSize: 22, color: C.text1, margin: 0, valign: "top" },
        text: "",
      },
    },
    {
      placeholder: {
        options: { name: "meta", type: "body", x: MX, y: 5.75, w: 8.0, h: 0.95, fontSize: 15, color: C.text2, margin: 0, valign: "bottom" },
        text: "",
      },
    },
  ],
});

pres.defineSlideMaster({
  title: "CF_STATEMENT",
  background: { color: C.accent1 },
  objects: [
    {
      placeholder: {
        options: { name: "eyebrow", type: "body", x: MX, y: 0.45, w: CW, h: 0.35, fontSize: 13, bold: true, color: C.background1, margin: 0 },
        text: "",
      },
    },
    {
      placeholder: {
        options: { name: "title", type: "title", x: MX, y: 0.8, w: CW, h: 0.95, fontSize: 30, bold: true, color: C.background1, margin: 0, valign: "top", align: "left", fontFace: "Cambria" },
        text: "",
      },
    },
  ],
  slideNumber: { x: 12.23, y: 6.95, w: 0.5, h: 0.3, fontSize: 11, color: HEX.lt1, align: "right" },
});

// ------------------------------------------------------------------ helpers
let current = { section: "" };
function section(title) {
  pres.addSection({ title });
  current.section = title;
}
function content(eyebrow, title, notes, master = "CF_CONTENT") {
  const s = pres.addSlide({ masterName: master, sectionTitle: current.section });
  s.addText(eyebrow, { placeholder: "eyebrow" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function card(s, name, x, y, w, h, fill = C.background2) {
  s.addShape(S.ROUNDED_RECTANGLE, { objectName: name, x, y, w, h, fill: { color: fill }, line: { type: "none" }, rectRadius: 0.12 });
}
function text(s, name, t, x, y, w, h, o = {}) {
  s.addText(t, { objectName: name, x, y, w, h, isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: C.text2, ...o });
}
function head(s, name, t, x, y, w, h, o = {}) {
  text(s, name, t, x, y, w, h, { fontFace: "+mj-lt", bold: true, color: C.text1, fontSize: 20, ...o });
}
function bigNumber(s, name, t, x, y, w, h, o = {}) {
  text(s, name, t, x, y, w, h, { fontFace: "+mj-lt", bold: true, color: C.accent1, fontSize: 44, valign: "middle", ...o });
}
function arrow(s, name, x, y, w = 0.45, h = 0.32) {
  s.addShape(S.RIGHT_ARROW, { objectName: name, x, y, w, h, fill: { color: C.accent4 }, line: { type: "none" } });
}
function chip(s, name, t, x, y, w, h, o = {}) {
  s.addShape(S.ROUNDED_RECTANGLE, { objectName: `${name} chip`, x, y, w, h, fill: { color: o.fill ?? C.background2 }, line: { color: o.line ?? C.accent5, width: 1 }, rectRadius: h / 2 });
  text(s, name, t, x, y, w, h, { align: "center", valign: "middle", fontSize: o.fontSize ?? 14, color: o.color ?? C.text1, bold: o.bold ?? false });
}
function bulletRuns(items, o = {}) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: o.space ?? 8 } }));
}
// One-series horizontal bar chart, best at the top, one highlighted bar (hex-only colour options).
function barChart(s, name, labels, values, highlight, x, y, w, h, title, fmt = "0.000") {
  s.addChart(pres.charts.BAR, [{ name: title, labels, values }], {
    objectName: name,
    x, y, w, h,
    barDir: "bar",
    catAxisOrientation: "maxMin",
    valAxisHidden: true,
    valGridLine: { style: "none" },
    catGridLine: { style: "none" },
    catAxisLineShow: false,
    chartColors: labels.map((l) => (l === highlight ? HEX.accent3 : HEX.accent2)),
    barGapWidthPct: 55,
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFormatCode: fmt,
    dataLabelColor: HEX.dk2,
    dataLabelFontSize: 11,
    dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk2,
    catAxisLabelFontSize: 12,
    catAxisLabelFontFace: "+mn-lt",
    showLegend: false,
    showTitle: true,
    title,
    titleColor: HEX.dk1,
    titleFontSize: 13,
    titleFontFace: "+mn-lt",
  });
}

// ------------------------------------------------------------------ 1. cover
section("Problem and approach");
{
  const s = pres.addSlide({ masterName: "CF_TITLE", sectionTitle: current.section });
  s.addText("Collaborative filtering · Final review", { placeholder: "eyebrow" });
  s.addText([{ text: "Red Box", options: { color: HEX.accent1 } }], { placeholder: "title" }); // the site's logo is all red
  s.addText("Collaborative Filtering for Personalized Movie/Product Recommendations using ML/DL Techniques", { placeholder: "subtitle" });
  s.addText("[Your name] · [Roll number]\nGuide: [Guide’s name] · [Department, College] · [Date]", { placeholder: "meta" });
  text(s, "Tagline", "Five models suggest.\nA sixth decides what you watch next.", 7.93, 5.75, 4.8, 0.95, {
    fontFace: "+mj-lt", italic: true, fontSize: 18, color: C.text2, align: "right", valign: "bottom",
  });
  s.addNotes(
    "Introduce yourself and the project in one sentence: we compared eleven collaborative-filtering algorithms fairly, built a hybrid that beats the best of them, and deployed it as a working web app.",
  );
}

// ------------------------------------------------------------------ 2. problem
{
  const s = content("The problem", "Too many choices, and no agreement on how to recommend", "Recommenders face four problems. The last one is the research question: published deep models often fail to beat well-tuned simple baselines when someone re-runs them fairly.");
  const items = [
    ["95.5%", "Sparsity", "of the MovieLens user–film matrix is empty"],
    ["65%", "Long tail", "of all ratings go to the 20% most popular films"],
    ["0", "Cold start", "ratings from a new user on their first visit"],
    ["?", "Contested evidence", "re-runs show tuned simple models often match deep ones"],
  ];
  const w = (CW - 3 * 0.3) / 4;
  items.forEach(([n, h, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `Problem card ${i + 1}`, x, 2.05, w, 2.9);
    bigNumber(s, `Problem number ${i + 1}`, n, x + 0.3, 2.3, w - 0.6, 0.9);
    head(s, `Problem heading ${i + 1}`, h, x + 0.3, 3.3, w - 0.6, 0.75, { fontSize: 18 });
    text(s, `Problem text ${i + 1}`, t, x + 0.3, 4.1, w - 0.6, 0.8, { fontSize: 14 });
  });
  text(s, "Problem takeaway", "So the project compares models fairly, looks beyond accuracy, and serves new users instantly.", MX, 5.45, CW, 0.5, { fontSize: 18, color: C.text1 });
}

// ------------------------------------------------------------------ 3. objectives + RQs
{
  const s = content("Objectives and research questions", "Seven objectives, four research questions", "Objectives O1 to O7 were all met; O6, deployment, is ready and needs free accounts to go live. The four research questions structure the results.");
  head(s, "Objectives heading", "Objectives", MX, 1.95, 5.6, 0.4, { fontSize: 18, color: C.accent1 });
  const objectives = [
    "Leakage-free data and evaluation pipeline",
    "Implement and tune 11 CF algorithms",
    "Compare accuracy and beyond-accuracy, with significance tests",
    "Propose a hybrid that beats the best single model",
    "Serve new users instantly; search films by description",
    "Deploy a full-stack app with explanations",
    "Check the findings on product data (Amazon)",
  ];
  objectives.forEach((t, i) => {
    const y = 2.45 + i * 0.6;
    chip(s, `Objective O${i + 1}`, `O${i + 1}`, MX, y, 0.62, 0.38, { fontSize: 12, bold: true, color: C.accent1 });
    text(s, `Objective text ${i + 1}`, t, MX + 0.8, y + 0.02, 4.9, 0.4, { fontSize: 15, color: C.text1 });
  });
  head(s, "RQ heading", "Research questions", 6.75, 1.95, 5.98, 0.4, { fontSize: 18, color: C.accent1 });
  const rqs = [
    "Do deep models beat well-tuned classic collaborative filtering?",
    "Does a two-stage hybrid beat the best single model, and why?",
    "How do models trade accuracy for diversity and coverage?",
    "Do the findings hold for products (Amazon) as well as films?",
  ];
  rqs.forEach((t, i) => {
    const y = 2.45 + i * 1.05;
    card(s, `RQ card ${i + 1}`, 6.75, y, 5.98, 0.85);
    text(s, `RQ label ${i + 1}`, `RQ${i + 1}`, 6.95, y, 0.8, 0.85, { fontFace: "+mj-lt", bold: true, fontSize: 20, color: C.accent1, valign: "middle" });
    text(s, `RQ text ${i + 1}`, t, 7.8, y, 4.75, 0.85, { fontSize: 15, color: C.text1, valign: "middle" });
  });
}

// ------------------------------------------------------------------ 4. protocol
{
  const s = content("Method", "One fair protocol for every model", "Every model sees the same split, ranks the whole catalogue, gets the same tuning budget and is tested exactly once. The test period is each user's newest 10%, so no model ever sees the future.");
  const steps = [
    ["Split by time", "Each user's oldest 80% trains, next 10% validates, newest 10% tests"],
    ["Rank everything", "Every unseen film is ranked: no sampled metrics"],
    ["Tune equally", "Optuna on validation, the same budget for each model"],
    ["Test once", "Paired Wilcoxon tests per user, Holm-corrected"],
  ];
  const w = (CW - 3 * 0.3) / 4;
  steps.forEach(([h, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `Step card ${i + 1}`, x, 2.0, w, 2.25);
    s.addShape(S.OVAL, { objectName: `Step number circle ${i + 1}`, x: x + 0.3, y: 2.25, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { type: "none" } });
    text(s, `Step number ${i + 1}`, String(i + 1), x + 0.3, 2.25, 0.5, 0.5, { align: "center", valign: "middle", bold: true, fontSize: 16, color: C.background1 });
    head(s, `Step heading ${i + 1}`, h, x + 0.3, 2.95, w - 0.6, 0.4, { fontSize: 18 });
    text(s, `Step text ${i + 1}`, t, x + 0.3, 3.4, w - 0.6, 0.8, { fontSize: 13 });
  });
  const datasets = [
    ["MovieLens-1M", "6,038 users × 3,458 films · 575,179 liked ratings (4 stars or more) · 77% of ratings share a second with another by the same user"],
    ["Amazon Reviews 2023 · Video Games", "42,045 users × 5,771 games after 5-core filtering · only 0.03% shared timestamps · the product benchmark"],
  ];
  const dw = (CW - 0.3) / 2;
  datasets.forEach(([h, t], i) => {
    const x = MX + i * (dw + 0.3);
    card(s, `Dataset card ${i + 1}`, x, 4.6, dw, 1.85);
    head(s, `Dataset name ${i + 1}`, h, x + 0.3, 4.85, dw - 0.6, 0.45, { fontSize: 18, color: C.accent1 });
    text(s, `Dataset text ${i + 1}`, t, x + 0.3, 5.35, dw - 0.6, 0.95, { fontSize: 14 });
  });
}

// ------------------------------------------------------------------ 5. models
{
  const s = content("Algorithms", "Eleven algorithms, one interface", "All eleven are implemented from their papers in NumPy, SciPy and PyTorch, behind one fit / score / recommend interface. No recommender library was used, which makes every model explainable in the viva.");
  const families = [
    ["Neighbourhood", [["MostPop", "the same popular list for everyone"], ["UserKNN", "people like you liked…"], ["ItemKNN", "films like the ones you liked"]]],
    ["Latent factor", [["BPR-MF", "rank liked films above the rest"], ["iALS", "weighted least squares on implicit data"]]],
    ["Linear autoencoder", [["EASE", "closed-form film-to-film weights"]]],
    ["Deep learning", [["NeuMF", "a neural interaction function"], ["Mult-VAE", "a variational autoencoder"], ["LightGCN", "graph convolution over users and films"]]],
    ["Sequential", [["Markov chain", "what usually follows your last film"], ["SASRec", "a Transformer over your history"]]],
  ];
  const w = (CW - 4 * 0.25) / 5;
  families.forEach(([fam, models], i) => {
    const x = MX + i * (w + 0.25);
    card(s, `Family card ${i + 1}`, x, 2.0, w, 4.0);
    text(s, `Family name ${i + 1}`, fam, x + 0.25, 2.2, w - 0.5, 0.4, { bold: true, fontSize: 14, color: C.accent1 });
    models.forEach(([m, idea], j) => {
      const y = 2.75 + j * 1.05;
      head(s, `Model ${m}`, m, x + 0.25, y, w - 0.5, 0.4, { fontSize: 18 });
      text(s, `Model idea ${m}`, idea, x + 0.25, y + 0.42, w - 0.5, 0.55, { fontSize: 12 });
    });
  });
  text(s, "Models footnote", "Implemented from the papers in NumPy, SciPy and PyTorch (CPU). No recommender library.", MX, 6.3, CW, 0.35, { fontSize: 13, color: C.accent4 });
}

// ------------------------------------------------------------------ 6. RQ1
section("What the comparison found");
{
  const s = content("RQ1 · Deep vs classic", "Tuned EASE beats every non-sequential deep model", "EASE solves one matrix inversion and trains in about two seconds. It is significantly more accurate than the graph network, the neural MF model and the variational autoencoder. This independently confirms the reproducibility studies.");
  barChart(
    s, "NDCG chart non-sequential",
    ["EASE", "BPR-MF", "LightGCN", "iALS", "UserKNN", "NeuMF", "Mult-VAE", "ItemKNN", "MostPop"],
    [0.081, 0.074, 0.0736, 0.0726, 0.0724, 0.0698, 0.0697, 0.0696, 0.0466],
    "EASE", MX, 1.95, 7.5, 4.85, "NDCG@10, MovieLens-1M test set (non-sequential models)",
  );
  card(s, "Training time card", 8.5, 2.05, 4.23, 2.0);
  bigNumber(s, "EASE training time", "1.9 s", 8.8, 2.25, 1.9, 0.8, { fontSize: 36 });
  text(s, "EASE training label", "EASE training", 8.8, 3.1, 1.9, 0.8, { fontSize: 13 });
  bigNumber(s, "LightGCN training time", "41 min", 10.7, 2.25, 1.9, 0.8, { fontSize: 36, color: C.accent2 });
  text(s, "LightGCN training label", "LightGCN training", 10.7, 3.1, 1.9, 0.8, { fontSize: 13 });
  text(s, "RQ1 significance", bulletRuns([
    "EASE vs LightGCN: +10.1%",
    "EASE vs NeuMF: +16.1%",
    "EASE vs Mult-VAE: +16.3%",
    "All p < 10⁻⁵ (Wilcoxon, Holm)",
  ]), 8.5, 4.35, 4.23, 1.6, { fontSize: 15, color: C.text1 });
  text(s, "RQ1 answer", "Answer to RQ1: no. Tune the simple baseline first.", 8.5, 6.15, 4.23, 0.5, { fontSize: 15, bold: true, color: C.accent1 });
}

// ------------------------------------------------------------------ 7. session artifact
{
  const s = content("Why sequential models win on MovieLens", "Most of the Transformer’s lead is the rating session", "96% of MovieLens test users were still in the bulk-rating session, where the website's display order predicts the next film. For users who come back days later, EASE is better than both sequential models. Shuffling SASRec's training histories removes 61% of its accuracy.");
  const hdr = { bold: true, color: HEX.accent4, fill: { color: HEX.lt1 } };
  const rows = [
    [
      { text: "First test rating came…", options: hdr },
      { text: "Users", options: { ...hdr, align: "right" } },
      { text: "EASE", options: { ...hdr, align: "right" } },
      { text: "Markov", options: { ...hdr, align: "right" } },
      { text: "SASRec", options: { ...hdr, align: "right" } },
      { text: "Hybrid", options: { ...hdr, align: "right" } },
    ],
    ["in the same session (< 1 h)", "5,774 (96%)", "0.081", "0.112", "0.168", "*0.183"],
    ["the same day (1–24 h)", "41", "*0.121", "0.080", "0.091", "0.108"],
    ["more than a day later", "219", "0.085", "0.062", "0.067", "*0.087"],
  ].map((r, i) =>
    i === 0
      ? r
      : r.map((c, j) => {
          const best = c.startsWith("*"); // best in the row: bold and red
          return {
            text: best ? c.slice(1) : c,
            options: { align: j === 0 ? "left" : "right", bold: best, color: best ? HEX.accent1 : HEX.dk1, fill: { color: HEX.lt2 } },
          };
        }),
  );
  s.addTable(rows, {
    objectName: "Session table",
    x: MX, y: 2.05, w: 7.6, colW: [2.9, 1.2, 0.9, 0.85, 0.85, 0.9],
    fontSize: 14, color: HEX.dk1, border: { type: "solid", pt: 1, color: HEX.lt1 }, rowH: 0.48, margin: 0.08,
  });
  text(s, "Session explanation", "NDCG@10 by when each test user's next rating came; the best in each row is bold red. In a bulk-rating session the site's display order predicts the next film, and only sequential models can learn that order.", MX, 4.35, 7.6, 1.0, { fontSize: 14 });
  card(s, "Ablation card", 8.6, 2.05, 4.13, 4.0);
  text(s, "Ablation label", "Order ablation: SASRec trained on shuffled histories", 8.9, 2.3, 3.6, 0.7, { fontSize: 14, bold: true, color: C.text1 });
  bigNumber(s, "Ablation numbers", "0.164 → 0.063", 8.9, 3.05, 3.6, 0.9, { fontSize: 34 });
  text(s, "Ablation result", bulletRuns(["−61% NDCG@10 (test set)", "falls below EASE (0.081)", "all of its lead came from order"]), 8.9, 4.1, 3.6, 1.6, { fontSize: 15, color: C.text1 });
  text(s, "Session takeaway", "A recommender is only as good as the way its offline data was collected.", MX, 5.6, 7.6, 0.6, { fontSize: 16, italic: true, color: C.text1 });
}

// ------------------------------------------------------------------ 8. hybrid pipeline
section("The proposed system");
{
  const s = content("Proposed system", "Red Box: five models suggest, a ranker decides", "Stage one: each model proposes its top 40 films; the union is about 123 films. Stage two: LightGBM LambdaRank, trained on the validation period, orders them using 41 features. Stage three: MMR re-ranking adds diversity on request, and every film gets a reason.");
  const gens = ["EASE", "SASRec", "Markov chain", "iALS (fold-in)", "MostPop"];
  gens.forEach((g, i) => chip(s, `Generator ${g}`, g, MX, 2.05 + i * 0.6, 2.3, 0.45, { fontSize: 14 }));
  arrow(s, "Arrow to candidates", 3.05, 3.1);
  const stages = [
    ["1 · Candidates", "Union of each model’s top 40: a median of 123 films", 3.65, 2.65],
    ["2 · Ranker", "LightGBM LambdaRank with 41 features: model scores and ranks, popularity, genre match, similarity to the last film, history length", 6.95, 3.05],
    ["3 · Diversity", "MMR re-ranking with a user-set λ, and a reason for every film", 10.65, 2.08],
  ];
  stages.forEach(([h, t, x, w], i) => {
    s.addShape(S.ROUNDED_RECTANGLE, { objectName: `Stage box ${i + 1}`, x, y: 2.05, w, h: 2.85, fill: { color: C.background2 }, line: { color: i === 1 ? C.accent1 : C.accent5, width: 1.5 }, rectRadius: 0.12 });
    head(s, `Stage heading ${i + 1}`, h, x + 0.25, 2.25, w - 0.5, 0.45, { fontSize: 16, color: C.accent1 });
    text(s, `Stage text ${i + 1}`, t, x + 0.25, 2.8, w - 0.5, 2.0, { fontSize: 14, color: C.text1 });
  });
  arrow(s, "Arrow to ranker", 6.4, 3.3);
  arrow(s, "Arrow to diversity", 10.1, 3.3);
  const notes = [
    ["No leakage", "models fit on the training period; the ranker learns from the later validation period"],
    ["Explained", "“Because you liked X” or “Matches your taste for Sci-Fi” on every film"],
    ["Deployable", "the app drops LightGCN (needs retraining) with no loss: 0.1789 vs 0.1790"],
  ];
  const w = (CW - 2 * 0.3) / 3;
  notes.forEach(([h, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `Pipeline note ${i + 1}`, x, 5.25, w, 1.4);
    head(s, `Pipeline note heading ${i + 1}`, h, x + 0.25, 5.4, w - 0.5, 0.4, { fontSize: 16 });
    text(s, `Pipeline note text ${i + 1}`, t, x + 0.25, 5.8, w - 0.5, 0.8, { fontSize: 13 });
  });
}

// ------------------------------------------------------------------ 9. gain (statement)
{
  const s = content("RQ2 · The proposed hybrid", "Red Box beats the best single model on both datasets", "Plus nine percent on MovieLens and eight point three on Amazon, both highly significant. The ranker relies most on SASRec's rank, then on EASE's taste score: it learns when to trust the sequence and when to fall back on long-term taste.", "CF_STATEMENT");
  const stats = [
    ["+9.0%", "MovieLens-1M", "NDCG@10 0.1643 (SASRec) → 0.1790"],
    ["+8.3%", "Amazon Video Games", "NDCG@10 0.0849 (SASRec) → 0.0919"],
  ];
  stats.forEach(([n, h, t], i) => {
    const x = MX + i * 6.2;
    text(s, `Gain number ${i + 1}`, n, x, 2.1, 5.8, 1.5, { fontFace: "+mj-lt", bold: true, fontSize: 96, color: C.background1, valign: "middle" });
    text(s, `Gain dataset ${i + 1}`, h, x, 3.7, 5.8, 0.5, { bold: true, fontSize: 20, color: C.background1 });
    text(s, `Gain detail ${i + 1}`, t, x, 4.2, 5.8, 0.5, { fontSize: 16, color: C.background1 });
  });
  text(s, "Gain significance", "Both p < 10⁻²⁶ (paired Wilcoxon, Holm-corrected). What the ranker leans on: SASRec’s rank (52% of its gain), then EASE’s taste score, history length and release year.", MX, 5.25, CW, 0.9, { fontSize: 16, color: C.background1 });
}

// ------------------------------------------------------------------ 10. RQ3
{
  const s = content("RQ3 · Beyond accuracy", "Variety is cheap, and accuracy doesn’t mean only hits", "MMR gives a smooth trade-off: at lambda 0.8 accuracy drops 4% while within-list diversity rises 16%. The app exposes this as the Diversity slider. The two most accurate systems also recommend a large share of the catalogue.");
  const hdr = { bold: true, color: HEX.accent4, fill: { color: HEX.lt1 }, align: "right" };
  const rows = [
    [{ text: "MMR λ", options: { ...hdr, align: "left" } }, { text: "NDCG@10", options: hdr }, { text: "Diversity@10", options: hdr }, { text: "Coverage@10", options: hdr }],
    ["1.0 (off)", "0.1790", "0.635", "0.532"],
    ["0.9", "0.1768", "0.686", "0.522"],
    ["0.8", "0.1716 (−4%)", "0.738 (+16%)", "0.511"],
    ["0.7", "0.1653", "0.784", "0.500"],
    ["0.5", "0.1467 (−18%)", "0.858 (+35%)", "0.480"],
  ].map((r, i) => (i === 0 ? r : r.map((c, j) => ({ text: c, options: { align: j === 0 ? "left" : "right", color: HEX.dk1, fill: { color: HEX.lt2 }, bold: i === 3 } }))));
  s.addTable(rows, { objectName: "MMR table", x: MX, y: 2.05, w: 5.9, colW: [1.45, 1.55, 1.5, 1.4], fontSize: 14, color: HEX.dk1, border: { type: "solid", pt: 1, color: HEX.lt1 }, rowH: 0.46, margin: 0.08 });
  text(s, "MMR caption", "The app’s Diversity slider sets λ.", MX, 5.0, 5.9, 0.4, { fontSize: 14, italic: true });
  barChart(s, "Coverage chart", ["SASRec", "Red Box hybrid", "LightGCN", "EASE", "MostPop"], [0.64, 0.532, 0.473, 0.232, 0.03], "Red Box hybrid", 6.9, 1.95, 5.83, 3.5, "Coverage@10: share of the catalogue recommended to anyone", "0%");
  text(s, "Coverage caption", "MostPop shows 3% of the catalogue; the two most accurate systems show more than half of it.", 6.9, 5.55, 5.83, 0.7, { fontSize: 14 });
}

// ------------------------------------------------------------------ 11. RQ4
{
  const s = content("RQ4 · Products", "On Amazon the session effect disappears; the conclusions hold", "Amazon purchases have no bulk-rating sessions. The Markov chain falls below EASE, which confirms that its MovieLens advantage was the session artifact. SASRec keeps a real lead: purchases have genuine order, like sequels and accessories. The hybrid is again best.");
  barChart(
    s, "NDCG chart Amazon",
    ["Red Box hybrid", "SASRec", "EASE", "ItemKNN", "UserKNN", "Markov chain", "LightGCN", "iALS", "BPR-MF", "MostPop"],
    [0.0919, 0.0849, 0.0706, 0.0647, 0.0629, 0.0618, 0.0586, 0.0585, 0.0321, 0.0215],
    "Red Box hybrid", MX, 1.95, 7.5, 4.85, "NDCG@10, Amazon Video Games test set",
  );
  const points = [
    ["0.03%", "of ratings share a timestamp: no bulk sessions"],
    ["−12%", "Markov chain vs EASE: its MovieLens lead was the artifact"],
    ["+20%", "SASRec vs EASE: purchases have real order"],
    ["+8.3%", "the hybrid is again the best system"],
  ];
  points.forEach(([n, t], i) => {
    const y = 2.05 + i * 1.18;
    card(s, `Amazon point ${i + 1}`, 8.5, y, 4.23, 1.02);
    bigNumber(s, `Amazon number ${i + 1}`, n, 8.75, y, 1.45, 1.02, { fontSize: 24 });
    text(s, `Amazon text ${i + 1}`, t, 10.25, y, 2.3, 1.02, { fontSize: 13, color: C.text1, valign: "middle" });
  });
}

// ------------------------------------------------------------------ 12. architecture
section("The deployed system");
{
  const s = content("System", "A real-time web app on free tiers", "The browser talks to a Next.js app on Vercel, which calls the FastAPI service on Hugging Face Spaces. Supabase stores accounts and ratings. The API recomputes recommendations from the user's current history on every request, so a rating changes the next list without retraining.");
  const boxes = [
    ["Browser", "any device"],
    ["Next.js web app", "Vercel · 8 pages"],
    ["FastAPI + models", "Hugging Face Spaces (Docker)"],
    ["Supabase", "Postgres + Auth, row-level security"],
  ];
  const bw = 2.55;
  const gap = (CW - 4 * bw) / 3;
  boxes.forEach(([h, t], i) => {
    const x = MX + i * (bw + gap);
    s.addShape(S.ROUNDED_RECTANGLE, { objectName: `Architecture box ${i + 1}`, x, y: 2.05, w: bw, h: 1.45, fill: { color: C.background2 }, line: { color: i === 2 ? C.accent1 : C.accent5, width: 1.5 }, rectRadius: 0.12 });
    head(s, `Architecture heading ${i + 1}`, h, x + 0.2, 2.25, bw - 0.4, 0.45, { fontSize: 17, align: "center" });
    text(s, `Architecture text ${i + 1}`, t, x + 0.2, 2.75, bw - 0.4, 0.6, { fontSize: 13, align: "center" });
    if (i < 3) arrow(s, `Architecture arrow ${i + 1}`, x + bw + (gap - 0.45) / 2, 2.62);
  });
  const stats = [
    ["34 ms", "median to compute a personal list, fresh from the user’s history"],
    ["0", "retraining after a rating: the very next list changes"],
    ["13 ms", "natural-language search over 3,458 film embeddings"],
    ["No PyTorch", "in the server: SASRec re-implemented in NumPy, MiniLM via ONNX"],
  ];
  const w = (CW - 3 * 0.3) / 4;
  stats.forEach(([n, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `System stat ${i + 1}`, x, 3.95, w, 2.3);
    bigNumber(s, `System number ${i + 1}`, n, x + 0.3, 4.15, w - 0.6, 0.85, { fontSize: n.length > 6 ? 28 : 40 });
    text(s, `System text ${i + 1}`, t, x + 0.3, 5.05, w - 0.6, 1.1, { fontSize: 14 });
  });
}

// ------------------------------------------------------------------ 13/14. screenshots
function screens(eyebrow, title, notes, left, right) {
  const s = content(eyebrow, title, notes);
  const w = (CW - 0.4) / 2;
  const h = w / 1.6; // screenshots are 1440 x 900
  [left, right].forEach(([file, alt, caption], i) => {
    const x = MX + i * (w + 0.4);
    s.addImage({ path: path.join(FIG, file), x, y: 2.0, w, h, altText: alt, objectName: `Screenshot ${file}`, rounding: false });
    text(s, `Caption ${file}`, caption, x, 2.0 + h + 0.2, w, 0.8, { fontSize: 14 });
  });
  return s;
}
screens(
  "The web app",
  "A personal feed that explains itself, and search by meaning",
  "Home: top picks from the hybrid, each with a reason, a diversity slider, and rows that re-rank the moment you rate a film. Search: describe what you want; your words and every film's keywords are compared as sentence embeddings.",
  ["home.png", "Red Box home feed with top picks, reasons and a diversity slider", "Home: top picks with a reason on every film, a diversity slider, instant re-ranking after each rating"],
  ["search.png", "Semantic search results for the query time travel", "Search: “time travel” finds Time Freak, Primer and 11.22.63 by their story and tags, not their titles"],
);
screens(
  "The web app",
  "Model Lab and results dashboard for the viva",
  "The Model Lab shows every algorithm's top ten for the same viewer, side by side, with an overlap heatmap and the MMR slider. The results page shows the offline experiments for MovieLens-1M, Amazon and the MovieLens-32M models the app serves, as charts with a table view and significance tests.",
  ["model-lab.png", "Model Lab comparing six models' top ten lists for one viewer", "Model Lab: one viewer, every model’s top 10 side by side, overlap heatmap, MMR slider"],
  ["results.png", "Results dashboard on the MovieLens-32M tab: the hybrid 15.9% ahead of SASRec", "Results: three datasets, including the live app’s MovieLens-32M models, with charts and p-values"],
);
screens(
  "The web app",
  "Every pick explains itself, and a group gets one list",
  "Film page: TreeSHAP splits the LightGBM ranker's score for this viewer into its signals, so the reason is the model's own arithmetic, not a template. Watch together: the hybrid scores a pooled candidate set for every member; the group chooses how to combine them: average, least misery or most pleasure.",
  ["why.png", "Film page breaking the ranker's score into signals such as EASE and SASRec", "Why this pick: the ranker’s score split into its signals with TreeSHAP"],
  ["watch-together.png", "Watch together list with each member's fit for every film", "Watch together: one list for a group, with each member’s fit; “nobody unhappy” is least misery"],
);
screens(
  "The web app",
  "New releases every day, in six languages",
  "The models' data ends in October 2023, but the app keeps up: every day the API finds new releases on Wikidata, in English, Hindi, Telugu, Tamil, Kannada and Malayalam, and recommends them from their Wikipedia story at once. 3,146 at build time. The interface switches between English, Telugu and Hindi.",
  ["new-films.png", "New English releases row with a filter for six languages", "New releases: 3,146 films newer than the models’ data, found daily, ranked from their story and ratings"],
  ["telugu.png", "Red Box home page in Telugu", "Three languages: English, Telugu and Hindi, with genres and reasons translated"],
);

// ------------------------------------------------------------------ new releases: arrival -> warm-up -> graduation
{
  const s = content("Keeping up with new films", "A new release gets smarter with every rating", "Three steps, none needing retraining until the last. Arrival: each day the API adds new releases and places them in the iALS taste space from their story. Warm-up: each rating re-estimates the film's factors, with the story's guess as the prior; that is collaborative topic regression. Graduation: once 20 app users like it, retraining makes it a regular film in every model. Measured on 848 catalogue films treated as new.");
  const cw = (CW - 0.3) / 2;
  card(s, "Release steps card", MX, 2.0, cw, 4.6);
  head(s, "Release steps heading", "From release to regular film", MX + 0.3, 2.2, cw - 0.6, 0.4, { fontSize: 18, color: C.accent1 });
  text(s, "Release steps list", bulletRuns([
    "Arrival, daily: Wikidata and Wikipedia give the story and poster; MiniLM text is mapped to iALS factors",
    "Warm-up, per rating: the story’s guess is the prior, each rating moves the film towards its real audience",
    "Graduation: 20 likes in the app, then retraining makes it a regular film in all five models",
  ], { space: 10 }), MX + 0.3, 2.75, cw - 0.6, 3.7, { fontSize: 16, color: C.text1 });
  barChart(
    s, "Warm-up chart",
    ["story only", "3 likes", "10 likes", "30 likes", "fully trained"],
    [0.0797, 0.0826, 0.0874, 0.0999, 0.1254],
    "30 likes",
    MX + cw + 0.3, 2.0, cw, 4.6,
    "NDCG@10 on films treated as new, by likes",
    "0.000",
  );
}

// ------------------------------------------------------------------ serving-time ML
{
  const s = content("Beyond the offline study", "Machine learning in every row of the live app", "These results are on MovieLens-32M, the data the app serves. Plus 15.9 percent over SASRec on the same 15,000 viewers, p below ten to the minus 86. The trending row and a new user's first picks come from a crowd forecast of the hybrid instead of popularity. New films nobody has rated are placed in the iALS space from their text.");
  const stats = [
    ["+15.9%", "hybrid over SASRec on MovieLens-32M, 15,000 viewers, p < 10⁻⁸⁶"],
    ["0.961", "trending NDCG@10 from the crowd forecast, against 0.537 for popularity"],
    ["17.2%", "hit@10 for a brand-new user from three genres, against 10.3% for popularity"],
    ["18.9%", "hit@10 on films treated as new, against 5.0% for text similarity alone"],
  ];
  const w = (CW - 3 * 0.3) / 4;
  stats.forEach(([n, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `Serving stat ${i + 1}`, x, 2.0, w, 2.2);
    bigNumber(s, `Serving number ${i + 1}`, n, x + 0.3, 2.15, w - 0.6, 0.8, { fontSize: 36 });
    text(s, `Serving text ${i + 1}`, t, x + 0.3, 3.0, w - 0.6, 1.1, { fontSize: 13 });
  });
  card(s, "Serving methods card", MX, 4.5, CW, 2.15);
  head(s, "Serving methods heading", "How", MX + 0.3, 4.65, CW - 0.6, 0.4, { fontSize: 16, color: C.accent1 });
  text(s, "Serving methods list", bulletRuns([
    "Crowd forecast: the hybrid’s top 10 for 4,000 viewers, summed by rank, overall and by genre taste",
    "New films: ridge map from MiniLM text to the 64 iALS factors (R² 0.18), refined by each rating",
    "Explanations: TreeSHAP contributions of the 41 ranker features, grouped by signal",
  ], { space: 4 }), MX + 0.3, 5.1, CW - 0.6, 1.45, { fontSize: 14, color: C.text1 });
}

// ------------------------------------------------------------------ 15. testing
{
  const s = content("Testing", "Tested from the maths to the browser", "197 automated tests run on every push, together with lint and type checks. A scripted browser walked through the whole user journey in Microsoft Edge, and a smoke test checks any deployment end to end.");
  const stats = [
    ["131", "ML tests: metrics, split, every model, hybrid, serving, forecast, new releases"],
    ["66", "API tests: endpoints, auth, user isolation, imports, releases, backups"],
    ["26", "browser journey steps, desktop and phone width, every language, no errors"],
    ["18/18", "smoke checks against the running stack"],
  ];
  const w = (CW - 3 * 0.3) / 4;
  stats.forEach(([n, t], i) => {
    const x = MX + i * (w + 0.3);
    card(s, `Test stat ${i + 1}`, x, 2.0, w, 1.95);
    bigNumber(s, `Test number ${i + 1}`, n, x + 0.3, 2.15, w - 0.6, 0.8, { fontSize: 40 });
    text(s, `Test text ${i + 1}`, t, x + 0.3, 3.0, w - 0.6, 0.85, { fontSize: 13 });
  });
  const cw = (CW - 0.3) / 2;
  card(s, "Guarantees card", MX, 4.25, cw, 2.4);
  head(s, "Guarantees heading", "What the tests guarantee", MX + 0.3, 4.4, cw - 0.6, 0.4, { fontSize: 16, color: C.accent1 });
  text(s, "Guarantees list", bulletRuns([
    "every metric equals a hand-computed value",
    "no user’s future is ever in training",
    "NumPy SASRec equals the PyTorch model",
    "live ranker features equal offline features",
  ], { space: 4 }), MX + 0.3, 4.85, cw - 0.6, 1.7, { fontSize: 14, color: C.text1 });
  const x2 = MX + cw + 0.3;
  card(s, "Caught card", x2, 4.25, cw, 2.4);
  head(s, "Caught heading", "What the tests caught", x2 + 0.3, 4.4, cw - 0.6, 0.4, { fontSize: 16, color: C.accent1 });
  text(s, "Caught list", bulletRuns([
    "train/serve skew in ranker features (latency 223 → 36 ms after the fix)",
    "noisy “hidden gems” → quality filter + content profile",
    "layout bugs seen in browser screenshots",
    "new films linked to the wrong article (a rainfall map as a poster)",
  ], { space: 4 }), x2 + 0.3, 4.85, cw - 0.6, 1.7, { fontSize: 14, color: C.text1 });
}

// ------------------------------------------------------------------ 16. conclusion
section("Conclusion");
{
  const s = content("Conclusion", "What we found, and what comes next", "Four findings: tuned simple models are hard to beat; MovieLens rewards rating order; combining models with a learned ranker helps on all three datasets; and research models can be served in real time cheaply, including for new users and new films. Future work: multi-seed statistics, a stricter cold-start test, online evaluation and more languages.");
  const cw = (CW - 0.3) / 2;
  card(s, "Findings card", MX, 2.0, cw, 4.6);
  head(s, "Findings heading", "Findings", MX + 0.3, 2.2, cw - 0.6, 0.4, { fontSize: 18, color: C.accent1 });
  text(s, "Findings list", bulletRuns([
    "Tuned EASE beats every non-sequential deep model, on both datasets",
    "MovieLens rewards rating order: shuffled, SASRec loses 61%",
    "Red Box: +9.0%, +8.3% and +15.9% over the best single model",
    "New releases join daily and learn from their first ratings",
    "Served on a CPU without retraining, new users and new films included",
  ], { space: 10 }), MX + 0.3, 2.75, cw - 0.6, 3.7, { fontSize: 16, color: C.text1 });
  const x2 = MX + cw + 0.3;
  card(s, "Future card", x2, 2.0, cw, 4.6);
  head(s, "Future heading", "Future scope", x2 + 0.3, 2.2, cw - 0.6, 0.4, { fontSize: 18, color: C.accent1 });
  text(s, "Future list", bulletRuns([
    "Several split seeds, confidence intervals, GPU tuning",
    "New films: retrain iALS without them for a stricter test",
    "Online evaluation from the logged impressions and clicks",
    "New releases in more languages, official TMDB artwork",
  ], { space: 10 }), x2 + 0.3, 2.75, cw - 0.6, 3.7, { fontSize: 16, color: C.text1 });
}

// ------------------------------------------------------------------ 17. thanks
{
  const s = pres.addSlide({ masterName: "CF_TITLE", sectionTitle: current.section });
  s.addText("Thank you", { placeholder: "eyebrow" });
  s.addText("Questions?", { placeholder: "title" });
  s.addText("Live demo, Model Lab and results dashboard available on request.", { placeholder: "subtitle" });
  s.addText("Demo: [URL after deployment] · Code: [GitHub link]\nReport: docs/report · Plan: docs/PROJECT_PLAN.md", { placeholder: "meta" });
  s.addNotes("Offer to show the live demo: onboarding, a rating that changes the list, semantic search and the Model Lab.");
}

// pptxgenjs writes Office's stock palette into the theme; put Red Box's colours there, so scheme colours
// (and PowerPoint's colour pickers) resolve to this deck's palette. Also mend one chart-axis element that
// pptxgenjs can write in a form PowerPoint "repairs" (crossesAt with autoZero).
async function writeThemeColors(file) {
  const fs = require("fs");
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const themePath = "ppt/theme/theme1.xml";
  let xml = await zip.file(themePath).async("string");
  for (const [slot, hex] of Object.entries(THEME.colors)) {
    const re = new RegExp(`<a:${slot}>[\\s\\S]*?</a:${slot}>`);
    if (!re.test(xml)) throw new Error(`theme slot ${slot} not found`);
    xml = xml.replace(re, `<a:${slot}><a:srgbClr val="${hex}"/></a:${slot}>`);
  }
  xml = xml.replace(/<a:clrScheme name="[^"]*"/, `<a:clrScheme name="${THEME.name}"`);
  zip.file(themePath, xml);
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const chart = await zip.file(name).async("string");
    zip.file(name, chart.replace(/<c:crossesAt val="autoZero"\/>/g, '<c:crosses val="autoZero"/>'));
  }
  // Re-pack files only: folder entries in the ZIP can stop cloud services from opening the deck.
  const out = new JSZip();
  for (const [name, entry] of Object.entries(zip.files)) {
    if (!entry.dir) out.file(name, await entry.async("uint8array"), { createFolders: false });
  }
  fs.writeFileSync(file, await out.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  } else {
    await writeThemeColors(OUT);
  }
  console.log(`wrote ${OUT}`);
})();
