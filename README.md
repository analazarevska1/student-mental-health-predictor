# MindCheck — Student Burnout Risk Predictor

A web application that predicts a student's burnout risk level (Low, Medium, High) from a short wellbeing survey, explains *why* using SHAP, and generates a personalised AI response with actionable, human-sounding advice.

Built as a final ML project covering the full pipeline: EDA → preprocessing → model comparison → explainability → a working full-stack web app.

---

## How it works

```
User fills survey (frontend)
        ↓
Flask receives answers (backend/app.py)
        ↓
Logistic Regression predicts risk level (backend/predict.py)
        ↓
SHAP explains which factors drove THIS prediction (backend/agent.py)
        ↓
LLM agent writes a personalised paragraph + tips + closing message
        ↓
Result shown to the user
```

---

## Project structure

```
mindcheck/
├── data/
│   └── raw_data.csv          # not included — see Dataset section below
├── notebooks/
│   ├── eda.ipynb             # exploration, correlation, feature importance
│   ├── preprocessing.ipynb   # cleaning, encoding, scaling, train/test split
│   └── model_training.ipynb  # trains & compares 5 models, saves the best one
├── models/
│   ├── best_model.pkl        # trained Logistic Regression (86.4% F1)
│   ├── scaler.pkl            # fitted StandardScaler
│   └── feature_names.pkl     # ordered feature list used by the model
├── backend/
│   ├── app.py                # Flask server, serves frontend + /predict route
│   ├── predict.py            # loads model, runs predictions
│   └── agent.py              # SHAP explainability + LLM-powered recommendations
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Dataset

[Student Mental Health and Burnout](https://www.kaggle.com/datasets/sharmajicoder/student-mental-health-and-burnout) — synthetic dataset, 150,000 student records, from Kaggle.

The raw CSV is not committed to this repo. To run the notebooks yourself, download it from the link above and place it at `data/raw_data.csv`.

---

## Setup

**1. Install dependencies** (using [uv](https://github.com/astral-sh/uv)):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt   # if generated, or:
pip install -e .
```

**2. Set up your API key**

```bash
cp .env.example .env
```

Edit `.env` and add your [OpenRouter](https://openrouter.ai/keys) API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**3. (Optional) Retrain the model**

The trained model is already included in `models/`. To retrain from scratch, download the dataset (see above) and run the three notebooks in order: `eda.ipynb` → `preprocessing.ipynb` → `model_training.ipynb`.

**4. Run the app**

```bash
uv run python backend/app.py
```

Open your browser at `http://localhost:5001`.

---

## Model

Five classifiers were trained and compared using 5-fold cross-validation on a 50,000-row stratified sample, with the winner retrained on the full dataset:

| Model               | CV F1 Mean |
|----------------------|------------|
| **Logistic Regression** | **0.8644** |
| XGBoost              | 0.8595     |
| Random Forest        | 0.8586     |
| SVM (RBF)            | 0.8586     |
| KNN (k=19)           | 0.8467     |

All five models scored within roughly 2 percentage points of each other, meaning the gap between them is closer to normal variation than a meaningful performance difference. With scores this close, the deciding factor became practicality: an unconstrained Random Forest initially scored marginally higher (0.8653) but produced a 1.2GB model file, which is impractical to deploy or commit to a repository. Constraining it to a deployable size dropped its score *below* Logistic Regression anyway.

**Logistic Regression** was selected as the final model — it matched or exceeded every other model's score, trains and predicts near-instantly, and the saved model is under 2KB. Full reasoning, including the Random Forest size-tuning experiment, is documented in `notebooks/model_training.ipynb`.

---

## AI-powered recommendations

Predictions alone aren't very useful without explaining *why*. For every prediction:

1. **SHAP** (`LinearExplainer`) calculates which features drove that specific student's result, ranked by impact
2. An **LLM agent** (via [Strands Agents](https://github.com/strands-agents/sdk-python) + [OpenRouter](https://openrouter.ai), free tier) receives the prediction, the SHAP values, and the raw survey answers
3. The agent generates three things in one call:
   - A short paragraph explaining *why*, in plain non-clinical language
   - 3-4 actionable, human-sounding tips — each tied to a specific factor
   - A closing message, tone-matched to the risk level

If the AI call fails for any reason (rate limits, network issues), the app falls back to a safe static message rather than breaking.

---

## Disclaimer

This tool is for educational purposes only. It is not a clinical diagnostic instrument. The dataset is synthetic. If you are struggling, please reach out to a real support service.