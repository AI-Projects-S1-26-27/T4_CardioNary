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

- **Header:** language and **Run screening batch**. The languages are the Chiron
  station's six: English, Hindi, Bengali, Telugu, Marathi and Tamil. The batch
  grades every reference case of both models and fills the accuracy cards.
- **System architecture:** the figure of the team's report, drawn live. Each block
  lights up while its stage runs. `cardio.predict.analyse_heart` and
  `lung.predict.analyse_xray` report their stages through `on_stage`. Between runs,
  the diagram marks the blocks for the step the user is on: choosing inputs,
  patient details, the decision, or the finished report.
- **Cardiac and pulmonary cards:**
  - case navigation over the reference cases (stepping to a case analyses it), or
    an upload; the heart sound and the X-ray of the current case are shown as soon
    as the page opens, before anything is analysed;
  - the verdict, confidence and match against the expected answer;
  - probability bars, and the late-fusion votes with their weights;
  - the heart sound and its spectrogram, or the X-ray input, its CLAHE view and the
    3 x 3 focus region;
  - timings, and the reporting threshold;
  - case details, with Chiron's clinical reference text.
- **Patient details:** optional. Empty fields stay empty; with no details the
  metadata vote is left out. For a male patient the pregnancy field is greyed out
  ("Not applicable") and the vote receives "not pregnant", as in the training data.
- **One decision per visit:** p = 1 - (1 - p_h)(1 - p_x). It refers when
  p >= tau = 1 / (1 + C_miss / C_refer), and can be exported as JSON.
- **Batch results, confidence history and case log** (CSV export). Records live
  only in the browser session.

## Reference cases

The dashboard is used in the team's presentation, so the cards show the held-out
cases (`Test_Data`, fold 0 of both models) that it answers correctly, listed in
`demo_cases.json`. `make_demo_cases.py` chooses them through the code of the
screening batch, with no patient details:

- the expected class at or above the default reporting threshold (60%), nothing
  withheld;
- the right visit decision at the default cost ratio (threshold 0.2). A diseased
  case alone is referred. A normal heart sound with a normal X-ray is not.

The first five cases of each card are diseased (for the heart: aortic stenosis,
mitral stenosis, mitral regurgitation, mitral valve prolapse, then aortic stenosis
again). After that, normal and diseased cases take turns. 51 of 53 heart sounds
and 112 of 120 X-rays are kept. The file lists the others with the reason. Because
the cases are chosen, the batch scores 100% on them. The held-out accuracy of each
model is printed under each batch figure.

Run `python app/make_demo_cases.py` again whenever a model or the fusion parameters
change. `tests/test_app.py` fails if a listed case is no longer answered correctly.

## Files

| File | What it holds |
|---|---|
| `app.py` | page, state and the runs |
| `dashboard_ui.py` | the stylesheet and markup ported from Chiron, and the architecture SVG |
| `dashboard_text.py` | CardioNary's strings in the six languages |
| `chiron_i18n.json` | Chiron's interface and clinical text in the six languages, exported unchanged from `gateway/static/i18n.js` |
| `demo_cases.json` | the reference cases the cards show, in order, and the ones left out |
| `make_demo_cases.py` | chooses `demo_cases.json` |

None of the non-English text has been reviewed by a native-speaking clinician.

## Rules kept

- No red anywhere; Streamlit's own red is themed amber.
- No emoji.
- No guessed patient data.
- Results are read through `core.contracts`.

## Tests

`tests/test_app.py`, `tests/test_app_no_usable_test.py` and
`tests/test_dashboard.py`. They set `CARDIONARY_STAGE_HOLD=0`, so the stages do
not pause.
