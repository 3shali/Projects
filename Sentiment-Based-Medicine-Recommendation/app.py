from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re

app = Flask(__name__, template_folder="templates")

# Load dataset
train_df = pd.read_csv("train.csv").dropna(subset=["condition", "review", "rating"])

# Load DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.half()  # ✅ Use half-precision (faster inference)

# Function for sentiment analysis
def analyze_sentiment_distilbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = probabilities[0][1].item()
    return "positive" if positive_score > 0.6 else "negative"

# ✅ Serve the Web UI
@app.route("/")
def home():
    return render_template("index.html")

# ✅ API Endpoint for Medicine Recommendation


@app.route("/recommend", methods=["GET"])
def recommend():
    condition = request.args.get("condition")

    if not condition:
        return jsonify({"error": "Please provide a condition."}), 400

    try:
        # ✅ Use regex-based filtering for faster matching
        pattern = re.compile(re.escape(condition), re.IGNORECASE)
        results = train_df[train_df['condition'].str.match(pattern, na=False)].copy()

        if results.empty:
            return jsonify({"message": "No medicines found for this condition."})

        # ✅ Optimize Sentiment Analysis Processing
        reviews = results["review"].tolist()
        sentiments = [analyze_sentiment_distilbert(review) for review in reviews]
        results["sentiment_label"] = [s[0] for s in sentiments]
        results["sentiment_score"] = [s[1] for s in sentiments]

        # ✅ Sort and return recommendations
        results = results.sort_values(by=["rating", "sentiment_score"], ascending=[False, False])
        recommendations = results[['drugName', 'rating', 'sentiment_label', 'review']].head(10).to_dict(orient="records")

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask on Render
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

