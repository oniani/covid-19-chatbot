#!/usr/bin/env python3

from flask import Flask, request, render_template
from generate_response import chatbot_response

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    """The index page management."""

    if request.method == "GET":
        return render_template("chatbot.html")

    # POST
    text = request.form.get("textarea")
    response = chatbot_response(text)
    return render_template("chatbot.html", response=response)
