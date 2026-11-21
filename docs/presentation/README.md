# Presentation

`RedBox_final_review.pptx`: the final-review deck. It has 17 slides with speaker notes, in the web app's
dark theme. Fonts are Cambria and Calibri, which ship with Office.

Before presenting, fill in the bracketed placeholders on the first and last slides:
- your name;
- roll number;
- guide;
- department and college;
- date;
- the demo URL;
- the GitHub link.

The slide numbers come from the report (`docs/report/11_results.md`, `10_testing.md`). To rebuild the deck
after changing `make_deck.js`, first install pptxgenjs in a scratch folder:

```bash
mkdir -p /tmp/deck && (cd /tmp/deck && npm install pptxgenjs@4.0.1)
NODE_PATH=/tmp/deck/node_modules node docs/presentation/make_deck.js
```

The demo-video script is in `docs/demo_script.md`.
