from dotenv import load_dotenv
load_dotenv()

import os
print("Key loaded:", os.environ.get("OPENROUTER_API_KEY", "NOT FOUND")[:15], "...")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict import predict_burnout, scale_input
from agent import get_ai_recommendation


app = Flask(__name__)
CORS(app)

LABEL_TO_CLASS = {"Low": 0, "Medium": 1, "High": 2}


# ── Serve frontend ──────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("../frontend", filename)


# ── Predict + AI recommendation ─────────────────────────
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    data = request.get_json()
    print("Received data:", data)

    if not data:
        return jsonify({"error": "No input data"}), 400

    # 1. Get prediction + the scaled array (needed again for SHAP)
    burnout_level, scaled_input = predict_burnout(data, return_scaled=True)
    predicted_class = LABEL_TO_CLASS[burnout_level]

    # 2. Ask the AI agent for a personalised message, backed by SHAP
    ai_result = get_ai_recommendation(
        burnout_level=burnout_level,
        scaled_input=scaled_input,
        raw_input=data,
        predicted_class=predicted_class
    )

    response = jsonify({
    "burnout_level": burnout_level,
    "ai_message": ai_result["message"],
    "tips": ai_result["tips"],          # ← this line is new
    "top_factors": ai_result["top_factors"],
    "closing": ai_result["closing"]
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
 