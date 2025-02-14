from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route("/similarity", methods=["POST"])
def calculate_similarity():
    data = request.get_json()
    original = data.get("original", "")
    user = data.get("user", "")

    if not original or not user:
        return jsonify({"error": "Both original and user texts are required."}), 400

    embedding1 = model.encode(original, convert_to_tensor=True)
    embedding2 = model.encode(user, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    return jsonify({"similarity": similarity})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
