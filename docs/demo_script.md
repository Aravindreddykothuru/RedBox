# Demo video script (about 5 minutes)

Record the screen at 1920×1080 with the local stack running (or the live site after deployment).

**Before recording:**
- Open each page once, so the first semantic search doesn't include the model's 1-second load.
- Have a Netflix history file ready (Account → Viewing activity → Download all). A small test file also
  works: `"Title","Date"` with `"RRR (Hindi)","20/11/2026"` and `"Dangal","12/11/2026"`.
- Use a new browser tab, so the opening animation plays.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:15 | Opening animation (new tab on the landing page) | "This is Red Box, a film recommender built on collaborative filtering." |
| 0:15–0:40 | Landing: scroll to "Trending now" | "Even this trending row comes from the models: the hybrid predicts what thousands of viewers will watch next, which beats plain popularity: NDCG 0.96 against 0.54." |
| 0:40–1:10 | Create an account → onboarding: pick Action, Drama; then "Import your history instead" → drop the Netflix file → preview → Add | "No need to rate anything. I import my Netflix history: titles are matched to the catalogue, TV episodes are left out, and I choose what to keep. A viewing counts as implicit feedback." |
| 1:10–1:50 | Home: billboard and rows; rate the top pick five stars | "My picks are ready, each with a reason. When I rate a film, the list re-ranks immediately: nothing is retrained, every model reads my history on each request, in about 70 milliseconds." |
| 1:50–2:10 | "New Indian films for you" row; click Tamil, then Malayalam | "These Telugu, Tamil, Kannada and Malayalam films are too new to have ratings, so no collaborative model can score them. A ridge mapping places each film's story in the iALS taste space, so it can still be recommended: that is item cold start." |
| 2:10–2:45 | Open the top pick: "How the ranker scored it for you" | "Why this pick? These bars are the exact contributions of each signal to the LightGBM ranker's score (TreeSHAP). Here SASRec, which reads my viewing order, pushed it up most; popularity pulled it down a little." |
| 2:45–3:15 | Watch together: add "Ravi" who loves Inception → "Nobody unhappy" → Find films | "Choosing for a group: the ranker scores the same films for each person, and least misery picks films nobody dislikes. Each bar shows the fit for one person." |
| 3:15–3:35 | Search → "Describe it" → "a slow-burn mystery in a small town" | "Search understands descriptions: a small Transformer embeds my words and each film's genres, keywords and Wikipedia summary, and compares them by meaning." |
| 3:35–3:55 | Header → language → తెలుగు; then back to English | "The whole interface, and even the models' reasons, are available in Telugu and Hindi." |
| 3:55–4:40 | Results: MovieLens-1M, then "MovieLens-32M (live app)" | "Offline, on held-out data: the hybrid beats the best single model by 9% on MovieLens-1M and by 16% on the app's MovieLens-32M models, with p < 10⁻⁸⁶. A simple linear model, EASE, beats every non-sequential deep model, and the Transformer's lead on MovieLens mostly comes from the order of the ratings." |
| 4:40–5:00 | Phone: install the app from the account menu, open it | "It installs like an app on a phone. Thank you." |

**Fallback.** If the live Space is asleep, show the local stack instead. The steps are in the README under
"Run the app locally".
