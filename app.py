# app.py
from flask import Flask, render_template, request, jsonify
from chatbot import get_chatbot_response

# Create a Flask application instance
app = Flask(__name__, template_folder='templates', static_folder='static')

# Route to serve the main chat page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle API calls from the front-end
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json.get("message")
    if user_message:
        chatbot_response = get_chatbot_response(user_message)
        return jsonify({"response": chatbot_response})
    return jsonify({"error": "No message provided"}), 400

# Run the application
if __name__ == "__main__":
    app.run(debug=True)