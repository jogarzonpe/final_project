import unittest

from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """
    Unit tests for the emotion_detector function from the EmotionDetection package.
    """

    def test_joy(self):
        """
        The dominant emotion for:
        "I am glad this happened"
        should be joy.
        """
        result = emotion_detector("I am glad this happened")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        """
        The dominant emotion for:
        "I am really mad about this"
        should be anger.
        """
        result = emotion_detector("I am really mad about this")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        """
        The dominant emotion for:
        "I feel disgusted just hearing about this"
        should be disgust.
        """
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness(self):
        """
        The dominant emotion for:
        "I am so sad about this"
        should be sadness.
        """
        result = emotion_detector("I am so sad about this")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear(self):
        """
        The dominant emotion for:
        "I am really afraid that this will happen"
        should be fear.
        """
        result = emotion_detector("I am really afraid that this will happen")
        self.assertIsNotNone(result)
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
