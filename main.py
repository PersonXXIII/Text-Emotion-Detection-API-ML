from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model
pipe_lr = joblib.load(open("Models/Emotion/custom_text_emotion.pkl", "rb"))

# Define the endpoint for emotion analysis
@app.route('/emotion', methods=['GET'])
def analyze_emotion():
    if 'note' not in request.args:
        return jsonify({"error": "Note parameter is required."}), 400

    note = request.args['note']
    result = predict_emotions(note)
    
    return jsonify({"emotion": result}), 200

# Function to predict emotion
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


if __name__ == '__main__':
    app.run(debug=True)
