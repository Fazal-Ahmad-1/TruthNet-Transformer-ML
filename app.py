from flask import Flask, request, jsonify
from transformers import pipeline, AutoConfig

app = Flask(__name__)

MODEL_NAME = "vikram71198/distilroberta-base-finetuned-fake-news-detection"

print("Loading pretrained DistilRoBERTa model...")

# Inspect the model's label configuration
config = AutoConfig.from_pretrained(MODEL_NAME)

print("Model label configuration:")
print(config.id2label)

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME
)

print("DistilRoBERTa model loaded successfully!")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Text is required"
        }), 400

    text = data["text"].strip()

    if not text:
        return jsonify({
            "error": "Text cannot be empty"
        }), 400

    result = classifier(text)[0]

    label = result["label"]
    confidence = result["score"] * 100

    # Print the raw model result so we can verify the mapping.
    print(f"Raw model result: {result}")

    # Temporary mapping.
    # We will verify this using the model's actual id2label configuration.
    if label in ["LABEL_1", "fake_news", "FAKE"]:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "model": "DistilRoBERTa"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "UP",
        "model": "DistilRoBERTa Fake News Detector"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )