from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained pipeline (vectorizer + model)
model = joblib.load('models/spam_model.pkl')

def predict_spam(message):
    pred = model.predict([message])[0]
    return "Spam" if pred == 1 else "Not Spam"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def api_data():
    data = request.get_json()
    message = data.get('text', '')
    result = predict_spam(message)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

