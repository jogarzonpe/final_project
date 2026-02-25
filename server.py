"""
Flask server for the Emotion Detection web application.

Static code analysis for this module can be executed with:
    pylint server.py
"""

from flask import Flask, request, render_template
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index():
    """
    Render the main page of the web application.
    """
    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET"])
def emotion_detector_route():
    """
    Handle the /emotionDetector endpoint.

    This endpoint receives a text string, calls the emotion_detector
    function from the EmotionDetection package, and returns a formatted
    response as plain text.

    If the dominant_emotion is None (for example, blank input or API
    returning status_code = 400), the response displays:
    "Invalid text! Please try again!".
    """
    text_to_analyze = request.args.get("textToAnalyze")
    result = emotion_detector(text_to_analyze)

    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    anger = result["anger"]
    disgust = result["disgust"]
    fear = result["fear"]
    joy = result["joy"]
    sadness = result["sadness"]
    dominant_emotion = result["dominant_emotion"]

    formatted_response = (
        "For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, "
        f"'fear': {fear}, 'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant_emotion}."
    )

    return formatted_response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
