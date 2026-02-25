"""
Emotion Detection Module

This module provides the emotion_detector function, which calls the
IBM Watson NLP EmotionPredict API to analyze text input and extract
emotion scores along with the dominant emotion.
"""

import json
import requests


def emotion_detector(text_to_analyze: str) -> dict:
    """
    Analyze the provided text using Watson NLP EmotionPredict API.

    Parameters:
        text_to_analyze (str): The input text to analyze.

    Returns:
        dict: A dictionary containing:
            - anger
            - disgust
            - fear
            - joy
            - sadness
            - dominant_emotion

        If the input is blank or the API request fails (e.g., status 400),
        all values are returned as None.
    """

    # Standard empty result structure (used for error cases)
    empty_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

    # Handle blank input
    if not text_to_analyze or not text_to_analyze.strip():
        return empty_result

    # Watson NLP EmotionPredict endpoint
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    # Required header
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    # Required JSON input format
    input_json = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        # Send POST request
        response = requests.post(url, json=input_json, headers=headers)

        # If response is not successful (e.g., 400)
        if response.status_code != 200:
            return empty_result

        # 1) Convert response text to dictionary using json library
        response_dict = json.loads(response.text)

        # 2) Extract the required emotion scores
        emotions = response_dict["emotionPredictions"][0]["emotion"]

        anger_score = emotions["anger"]
        disgust_score = emotions["disgust"]
        fear_score = emotions["fear"]
        joy_score = emotions["joy"]
        sadness_score = emotions["sadness"]

        # Dictionary to help compute the dominant emotion
        emotion_scores = {
            "anger": anger_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "joy": joy_score,
            "sadness": sadness_score,
        }

        # 3) Find the dominant emotion (emotion with the highest score)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        # 4) Return the output in the required format
        result = {
            "anger": anger_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "joy": joy_score,
            "sadness": sadness_score,
            "dominant_emotion": dominant_emotion,
        }

        return result

    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
        # In case of any error, return empty result
        return empty_result
