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
    Analyze text using Watson NLP EmotionPredict API.
    Returns emotion scores and dominant emotion.

    For blank input or if the API returns status_code = 400,
    the function returns a dictionary where all values are None.
    """

    # Standard empty result structure
    empty_result = {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }

    # Handle blank input from user (no text entered)
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
        # Call Watson NLP API
        response = requests.post(url, json=input_json, headers=headers)

        # 🔴 Task 7 requirement:
        # If status_code = 400, return dictionary with all values = None
        if response.status_code == 400:
            return empty_result

        # Any other non-200 status → also treat as invalid
        if response.status_code != 200:
            return empty_result

        # Convert response text to dictionary
        response_dict = json.loads(response.text)

        # Extract required emotion scores
        emotions = response_dict["emotionPredictions"][0]["emotion"]

        anger_score = emotions["anger"]
        disgust_score = emotions["disgust"]
        fear_score = emotions["fear"]
        joy_score = emotions["joy"]
        sadness_score = emotions["sadness"]

        # Compute dominant emotion (highest score)
        emotion_scores = {
            "anger": anger_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "joy": joy_score,
            "sadness": sadness_score,
        }

        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        # Return in the required format
        return {
            "anger": anger_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "joy": joy_score,
            "sadness": sadness_score,
            "dominant_emotion": dominant_emotion,
        }

    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
        # On any error, return all None
        return empty_result
