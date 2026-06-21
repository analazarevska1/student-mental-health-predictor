"""
agent.py — AI-powered personalised recommendation engine

Uses SHAP to explain WHY the model predicted a given risk level for this
specific student, then passes that explanation to an LLM agent (via Strands +
OpenRouter) which writes:
  1. A short paragraph explaining WHY (referencing SHAP factors)
  2. 3-4 actionable, human-sounding tips
  3. A short closing message, tone-matched to the risk level

Run is triggered from app.py after predict_burnout() returns a result.
"""

import os
import pickle
import numpy as np
import shap

from strands import Agent, tool
from strands.models.openai import OpenAIModel

# ── Paths ────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE, "../models/best_model.pkl")
FEATURES_PATH = os.path.join(BASE, "../models/feature_names.pkl")
X_TRAIN_PATH = os.path.join(BASE, "../models/shap_background.npy")
X_train_background = np.load(X_TRAIN_PATH)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    FEATURES = pickle.load(f)

# ── SHAP explainer ───────────────────────────────────────────────────────
explainer = shap.LinearExplainer(model, X_train_background[:200])


LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def get_shap_explanation(scaled_input: np.ndarray, predicted_class: int) -> list[dict]:
    """
    Calculates SHAP values for a single prediction and returns the
    top contributing features sorted by impact, for the predicted class.
    """
    shap_values = explainer.shap_values(scaled_input)

    if isinstance(shap_values, list):
        # Multi-class as a list of arrays, one per class (older SHAP / TreeExplainer style)
        class_shap = shap_values[predicted_class][0]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # Multi-class as one 3D array: (1, n_features, n_classes)
        class_shap = shap_values[0, :, predicted_class]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        # Single set of values, no class dimension (common with LinearExplainer)
        class_shap = shap_values[0]

    else:
        raise ValueError(f"Unexpected SHAP output shape: {np.shape(shap_values)}")

    contributions = [
        {"feature": feat, "impact": round(float(val), 4)}
        for feat, val in zip(FEATURES, class_shap)
    ]
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:5]


# ── Model setup — same pattern as CODI ──────────────────────────────────
model_llm = OpenAIModel(
    model_id="openai/gpt-oss-120b:free",
    params={"max_tokens": 700, "temperature": 0.6},
    client_args={
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "base_url": "https://openrouter.ai/api/v1"
    }
)


@tool
def interpret_shap_factors(factors: str) -> str:
    """
    Takes a formatted string describing the top SHAP factors driving this
    student's burnout prediction and returns a short structured summary
    identifying which factors are increasing risk vs protecting against it.
    """
    return f"Factors received for interpretation:\n{factors}"


# ── Agent definition ────────────────────────────────────────────────────
agent = Agent(
    model=model_llm,
    tools=[interpret_shap_factors],
    system_prompt="""You are a warm, supportive student wellbeing assistant.

You receive a student's burnout risk prediction (Low, Medium, or High), their
survey answers, and SHAP values showing exactly which factors drove THIS
SPECIFIC prediction, ranked by impact.

When referring to these features in your writing, always use these natural,
non-clinical terms instead of the raw names:
- stress_level → "stress"
- anxiety_score → "anxiety" or "feeling on edge"
- depression_score → "low mood" or "motivation"
- sleep_hours → "sleep"
- exam_pressure → "exam pressure"
- physical_activity → "movement" or "activity levels"
- social_support → "connection with others"
- financial_stress → "money stress"
- family_expectation → "family pressure"
- study_hours_per_day → "study time"
Never use the raw underscore_case feature names anywhere in your response.

Your response has THREE parts, in this exact order.

═══════════════════════════════════════════════════
PART 1 — Explanatory paragraph
═══════════════════════════════════════════════════
Write ONE short, personal paragraph (4-6 sentences, under 100 words)
explaining WHY the model reached this result for this specific student.

Rules:
- Speak directly to the student ("you", "your")
- Reference their SPECIFIC top 2-3 factors by name (using the natural terms
  above), explain in plain language what they mean and why they matter
- Be encouraging and non-clinical — you are not a therapist, you are a
  supportive guide
- Never use the word "burnout" more than once
- Do not list bullet points — write flowing, natural prose
- Do not repeat the risk level back mechanically ("Your risk level is High")
  — weave it in naturally
- If risk is Low, focus on reinforcing what they're doing right
- Do NOT include any action tips in this paragraph — save those for Part 2

After the paragraph, write the marker ---TIPS--- on its own line.

═══════════════════════════════════════════════════
PART 2 — Actionable tips
═══════════════════════════════════════════════════
Write 3-4 tips, one per line, each starting with a dash. Each tip should be
15-50 words long.

Write tips the way a close friend who actually knows this student would text
them — casual, a little blunt sometimes, no "try doing X for Y minutes"
formulas, no explaining the psychological mechanism behind why something
works. Just say the thing plainly, like advice over coffee, not a wellness
app notification.

Each tip must reference WHY it matters for this student — connect it back to
one of their specific factors. Do not re-explain the factors in detail — the
paragraph already did that, just reference briefly while giving the action.

Address a DIFFERENT factor in each tip — don't repeat the same theme twice.

Good tip: "Your sleep's honestly fine, so don't stress about that one — but
exam pressure is clearly wearing on you. Maybe block off Sunday afternoons
as a no-studying zone, even if it feels wrong at first."

Good tip: "You said you barely see your friends right now. Even texting one
person to grab coffee this week would probably do more for you than
anything else on this list."

Bad tip: "Schedule a brief daily worry window to write down anxious
thoughts." (too clinical)

Bad tip: "Block out the last 30 minutes of your study blocks for something
that's just for you — even 10 minutes of music can stop stress from
compounding." (still sounds like an app notification, too neat and
mechanism-explainy)

After the tips, write the marker ---CLOSING--- on its own line.

═══════════════════════════════════════════════════
PART 3 — Closing message
═══════════════════════════════════════════════════
Write ONE short closing message (1-2 sentences, under 35 words). Tone
depends on the risk level:

- If risk is Low: a light, encouraging line reinforcing that they're doing
  well. Do NOT mention seeking help or not being alone — that would feel
  confusing or alarming for someone who is doing fine.

- If risk is Medium: acknowledge they're carrying a lot right now, and
  gently note that reaching out to someone — a friend, family, or otherwise —
  can help. Keep it warm, not alarming.

- If risk is High: be warm and direct. Let them know what they're feeling
  matters, they don't have to handle it alone, and encourage them to talk to
  someone — a friend, family member, or their university's counselling
  service. Do not sound clinical or scripted.

Rules for this message regardless of risk level:
- Do not use dramatic language ("everything happens for a reason", "you are
  not broken", "this too shall pass")
- Do not diagnose or use clinical terms
- Do not repeat the word "alone" more than once
- Keep it simple and sincere, like something a caring friend would say at
  the end of a conversation, not a wellness app's notification
"""
)


def extract_text(result) -> str:
    """Same robust extractor used in the CODI project."""
    if hasattr(result, 'message'):
        msg = result.message
        if isinstance(msg, dict):
            content = msg.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get('text', '')
                    for block in content
                    if isinstance(block, dict) and 'text' in block
                )
    if isinstance(result, dict):
        content = result.get('content', '')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get('text', '')
                for block in content
                if isinstance(block, dict) and 'text' in block
            )
    return str(result)


FALLBACK_CLOSING = {
    "Low":    "Keep doing what you're doing — it's working.",
    "Medium": "You don't have to carry everything alone — reaching out to someone can really help.",
    "High":   "What you're feeling matters, and support is always available if you want to talk to someone.",
}


def get_ai_recommendation(burnout_level: str, scaled_input: np.ndarray,
                           raw_input: dict, predicted_class: int) -> dict:
    """
    Main entry point called from app.py.

    Returns:
    {
        "message": "<AI generated paragraph explaining WHY>",
        "tips": ["<tip 1>", "<tip 2>", ...],
        "closing": "<short risk-aware closing message>",
        "top_factors": [{"feature": ..., "impact": ...}, ...]
    }
    """
    top_factors = get_shap_explanation(scaled_input, predicted_class)

    factors_text = "\n".join(
        f"- {f['feature']}: {'increases' if f['impact'] > 0 else 'decreases'} "
        f"risk (impact score: {abs(f['impact'])})"
        for f in top_factors
    )

    prompt = f"""Student's predicted burnout risk: {burnout_level}

Top factors driving this specific prediction (from SHAP analysis):
{factors_text}

Raw survey answers: {raw_input}

Write your response now, following your system instructions exactly —
the paragraph, ---TIPS---, the tips, ---CLOSING---, then the closing message."""

    try:
        result = agent(prompt)
        raw_message = extract_text(result)

        # Split into paragraph / tips / closing
        parts = raw_message.split('---TIPS---')
        paragraph = parts[0].strip()

        remainder = parts[1] if len(parts) > 1 else ""
        tips_parts = remainder.split('---CLOSING---')
        tips_block = tips_parts[0].strip()
        closing = tips_parts[1].strip() if len(tips_parts) > 1 else FALLBACK_CLOSING.get(burnout_level, "")

        tips = [
            line.lstrip("-").strip()
            for line in tips_block.splitlines()
            if line.strip()
        ]

        if not tips:
            tips = ["Consider focusing on your top risk factor this week."]
        if not closing:
            closing = FALLBACK_CLOSING.get(burnout_level, "")

    except Exception as e:
        paragraph = (
            f"We couldn't generate a personalised AI message right now "
            f"({str(e)}). Based on your results, your top factor was "
            f"{top_factors[0]['feature']} — consider focusing there first."
        )
        tips = ["Try reviewing your top risk factors below for guidance."]
        closing = FALLBACK_CLOSING.get(burnout_level, "")

    return {
        "message": paragraph,
        "tips": tips,
        "closing": closing,
        "top_factors": top_factors
    }