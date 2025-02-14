import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://studywave-sr.netlify.app"}})

# Load model once at startup to save memory
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Text Similarity API!"})

@app.route("/similarity", methods=["POST"])
def calculate_similarity():
    try:
        data = request.get_json()
        original = data.get("original", "").strip()
        user = data.get("user", "").strip()

        if not original or not user:
            return jsonify({"error": "Both original and user texts are required."}), 400

        # Compute similarity
        embedding1 = model.encode(original, convert_to_tensor=True)
        embedding2 = model.encode(user, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

        return jsonify({"similarity": similarity})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
