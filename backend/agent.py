"""
agent.py — AI-powered personalised recommendation engine

Uses SHAP to explain WHY the model predicted a given risk level for this
specific student, then passes that explanation to an LLM agent (via Strands +
OpenRouter) which writes a short paragraph, tips, and a closing message.

SAFETY DESIGN:
Concerning values (very low sleep, very low activity, very high
stress/anxiety/mood scores) are flagged in Python BEFORE the prompt is built.
The LLM is explicitly told which factors are flagged and is given a fixed,
pre-approved direction for those — it does not get creative freedom on
whether low sleep or low activity is good or bad. This avoids relying on the
LLM to consistently self-police nuanced safety rules across every response.
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
BG_PATH       = os.path.join(BASE, "../models/shap_background.npy")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    FEATURES = pickle.load(f)

X_train_background = np.load(BG_PATH)

# ── SHAP explainer ───────────────────────────────────────────────────────
explainer = shap.LinearExplainer(model, X_train_background)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def get_shap_explanation(scaled_input: np.ndarray, predicted_class: int) -> list[dict]:
    """Calculates SHAP values for a single prediction, returns top 5 factors."""
    shap_values = explainer.shap_values(scaled_input)

    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        class_shap = shap_values[0]
    else:
        raise ValueError(f"Unexpected SHAP output shape: {np.shape(shap_values)}")

    contributions = [
        {"feature": feat, "impact": round(float(val), 4)}
        for feat, val in zip(FEATURES, class_shap)
    ]
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:5]


# ── Safety net — Python decides direction, not the LLM ──────────────────
# For each factor, define what counts as "concerning" and the ONE fixed
# direction the LLM must use. The LLM cannot override this.
SAFETY_RULES = {
    "sleep_hours":        {"concerning": lambda v: v < 5,  "direction": "needs to INCREASE — this is low"},
    "physical_activity":  {"concerning": lambda v: v < 3,  "direction": "needs to INCREASE — this is low"},
    "social_support":     {"concerning": lambda v: v < 3,  "direction": "needs to INCREASE — this is low"},
    "stress_level":       {"concerning": lambda v: v > 7,  "direction": "needs to DECREASE — this is high"},
    "anxiety_score":      {"concerning": lambda v: v > 7,  "direction": "needs to DECREASE — this is high"},
    "depression_score":   {"concerning": lambda v: v > 7,  "direction": "needs to DECREASE — this is high"},
    "exam_pressure":      {"concerning": lambda v: v > 8,  "direction": "needs to DECREASE — this is high"},
    "financial_stress":   {"concerning": lambda v: v > 8,  "direction": "needs to DECREASE — this is high"},
    "family_expectation": {"concerning": lambda v: v > 8,  "direction": "needs to DECREASE — this is high"},
}


def build_factor_lines(top_factors: list[dict], raw_input: dict) -> str:
    """
    For each top factor, build one line for the prompt. If the value is
    flagged as concerning, the line includes a fixed, non-negotiable
    direction the LLM must follow — it is not left to interpret SHAP's
    sign itself.
    """
    lines = []
    for f in top_factors:
        feat = f["feature"]
        value = raw_input.get(feat, "unknown")
        rule = SAFETY_RULES.get(feat)

        if rule and isinstance(value, (int, float)) and rule["concerning"](value):
            lines.append(
                f"- {feat}: actual value = {value}. FLAGGED: {rule['direction']}. "
                f"You MUST encourage this direction. Do not normalize, soften, "
                f"or treat the current value as acceptable."
            )
        else:
            lines.append(f"- {feat}: actual value = {value} (not flagged, use your judgment)")

    return "\n".join(lines)


# ── Model setup ───────────────────────────────────────────────────────────
model_llm = OpenAIModel(
    model_id="openai/gpt-oss-120b:free",
    params={"max_tokens": 600, "temperature": 0.6},
    client_args={
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "base_url": "https://openrouter.ai/api/v1"
    }
)


@tool
def interpret_shap_factors(factors: str) -> str:
    """Takes formatted SHAP factor info and returns it for interpretation."""
    return f"Factors received for interpretation:\n{factors}"


# ── Agent — short, focused system prompt ─────────────────────────────────
agent = Agent(
    model=model_llm,
    tools=[interpret_shap_factors],
    system_prompt="""You are a warm, casual student wellbeing assistant — like a
supportive older friend, not a clinician.

You will be given a risk level (Low/Medium/High) and a list of factors, each
already labeled either FLAGGED (with a required direction) or not flagged.

Write three parts:

1. A short paragraph (under 100 words) explaining why this risk level, in
plain language. Only reference factors from the list given. For FLAGGED
factors, follow the required direction exactly. Never say a flagged value
is "decent," "solid," "good," or something to "protect" or "maintain."

Write ---TIPS--- on its own line.

2. 3-4 short tips (15-40 words each), one per line starting with "-". Each
tip must address a different factor. For FLAGGED factors, the tip must
clearly push in the required direction (e.g. tell them to sleep more, never
to "protect" short sleep). For non-flagged factors, use your judgment for
genuinely supportive, specific, casual advice.

Write ---CLOSING--- on its own line.

3. One short closing line (under 30 words). Low risk: light encouragement,
no mention of seeking help. Medium: acknowledge the load, gently suggest
reaching out to someone. High: warm and direct, encourage talking to
someone — friend, family, or counselling service.

Style: casual and human, like a text from a friend, not a wellness app.
No therapy jargon. Use natural terms (stress, anxiety, low mood, sleep,
movement, money stress, family pressure) instead of raw feature names.
"""
)


def extract_text(result) -> str:
    """Robust text extractor for Strands AgentResult."""
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
    print("TOP FACTORS:", top_factors)
    factor_lines = build_factor_lines(top_factors, raw_input)

    prompt = f"""Risk level: {burnout_level}

Factors:
{factor_lines}

Write your response now: paragraph, ---TIPS---, tips, ---CLOSING---, closing line."""

    try:
        result = agent(prompt)
        raw_message = extract_text(result)

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