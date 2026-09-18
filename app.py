"""
Flask web server for the FAQ Chatbot.
Run with: python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, request, jsonify, render_template
from chatbot import FAQChatbot

app = Flask(__name__)
bot = FAQChatbot("faqs.csv")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    user_message = data.get("message", "")

    answer, matched_question, score = bot.get_response(user_message)

    return jsonify({
        "answer": answer,
        "matched_question": matched_question,
        "score": round(score, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
