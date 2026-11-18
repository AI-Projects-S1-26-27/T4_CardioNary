# app: the CardioNary screening dashboard

```
streamlit run app/app.py
```

Run it from the repository root, so `.streamlit/config.toml` (the dark theme) is
read. The public demo of branch `deploy-streamlit-cloud` runs at
https://cardionary.streamlit.app/.

## What is on the page

The layout and most features come from the Chiron screening station
(`gateway/static`), restyled without red:

- **Header:** language (English, Hindi) and **Run screening batch**. The batch
  grades every reference case of both models and fills the accuracy cards.
- **System architecture:** the figure of the team's report, drawn live. Each block
  lights up while its stage runs. `cardio.predict.analyse_heart` and
  `lung.predict.analyse_xray` report their stages through `on_stage`. Between runs,
  the diagram marks the blocks for the step the user is on: choosing inputs,
  patient details, the decision, or the finished report.
- **Cardiac and pulmonary cards:**
  - case navigation over the reference cases (stepping to a case analyses it), or
    an upload;
  - the verdict, confidence and match against the expected answer;
  - probability bars, and the late-fusion votes with their weights;
  - the heart sound and its spectrogram, or the X-ray input, its CLAHE view and the
    3 x 3 focus region;
  - timings, and the reporting threshold;
  - case details, with Chiron's clinical reference text.
- **Patient details:** optional. Empty fields stay empty; with no details the
  metadata vote is left out.
- **One decision per visit:** p = 1 - (1 - p_h)(1 - p_x). It refers when
  p >= tau = 1 / (1 + C_miss / C_refer), and can be exported as JSON.
- **Batch results, confidence history and case log** (CSV export). Records live
  only in the browser session.

## Files

| File | What it holds |
|---|---|
| `app.py` | page, state and the runs |
| `dashboard_ui.py` | the stylesheet and markup ported from Chiron, and the architecture SVG |
| `dashboard_text.py` | CardioNary's English and Hindi strings |
| `chiron_i18n.json` | Chiron's interface and clinical text, exported unchanged from `gateway/static/i18n.js` |

The Hindi text has not been reviewed by a native-speaking clinician.

## Rules kept

- No red anywhere; Streamlit's own red is themed amber.
- No emoji.
- No guessed patient data.
- Results are read through `core.contracts`.

## Tests

`tests/test_app.py`, `tests/test_app_no_usable_test.py` and
`tests/test_dashboard.py`. They set `CARDIONARY_STAGE_HOLD=0`, so the stages do
not pause.
