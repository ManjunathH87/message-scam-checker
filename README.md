# Message Scam Checker — Streamlit

A simple, mobile-friendly app combining a saved scikit-learn pipeline, rule-based checks, and optional Gemini explanations.

## Files required

Place these files in the same folder:
- `app.py`
- `requirements.txt`
- `scam_detection_model.joblib` — saved TF-IDF + Logistic Regression Pipeline

The saved pipeline should have steps named `tfidf` and `classifier`, and classes named `scam` and `safe`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configure Gemini

Set `GEMINI_API_KEY` as an environment variable.

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

macOS/Linux:
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

Alternatively create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Do not commit your API key or `secrets.toml` to a public repository.

## Add your model file

Download `/content/scam_detection_model.joblib` from Colab and put it beside `app.py`. You can set `SCAM_MODEL_PATH` to another location if needed.

## Notes

- The app still performs ML + rule detection if Gemini is not configured.
- Gemini model defaults to `gemini-3.8-flash`. If model availability changes for your account, set `GEMINI_MODEL` to a model supported by your Gemini API project.
- URL and brand-lookalike rules are heuristics and are not a substitute for a maintained threat-intelligence feed.
- A `Safe` result is not a guarantee. Do not use this as the sole basis for blocking messages or taking financial action.
