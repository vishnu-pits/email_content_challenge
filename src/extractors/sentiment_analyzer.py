from textblob import TextBlob
import spacy
from typing import Dict, List
import logging
from collections import defaultdict
import numpy as np


class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_weights = {
            'subject': 0.3,
            'body': 0.5,
            'signature': 0.2
        }

    def analyze_sentiment(self, email_data: Dict) -> Dict:
        """
        Analyze sentiment of email components and return detailed sentiment analysis.
        Returns dict with overall score and component-wise analysis.
        """
        try:
            # Analyze different components
            subject_sentiment = self._analyze_text(email_data.get('subject', ''))
            body_sentiment = self._analyze_text(email_data.get('body', ''))
            signature_sentiment = self._analyze_text(email_data.get('signature', ''))

            # Calculate weighted sentiment
            overall_sentiment = (
                    subject_sentiment['score'] * self.sentiment_weights['subject'] +
                    body_sentiment['score'] * self.sentiment_weights['body'] +
                    signature_sentiment['score'] * self.sentiment_weights['signature']
            )

            # Extract emotional indicators
            emotions = self._detect_emotions(email_data.get('body', ''))

            return {
                'overall_score': overall_sentiment,
                'components': {
                    'subject': subject_sentiment,
                    'body': body_sentiment,
                    'signature': signature_sentiment
                },
                'emotions': emotions,
                'confidence': self._calculate_confidence(
                    subject_sentiment, body_sentiment, signature_sentiment
                )
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'overall_score': 0, 'components': {}, 'emotions': [], 'confidence': 0}

    def _analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a text segment."""
        if not text:
            return {'score': 0, 'magnitude': 0, 'polarity': 'neutral'}

        blob = TextBlob(text)

        # Get base sentiment
        sentiment_score = blob.sentiment.polarity
        sentiment_magnitude = abs(sentiment_score)

        # Determine polarity
        if sentiment_score > 0.1:
            polarity = 'positive'
        elif sentiment_score < -0.1:
            polarity = 'negative'
        else:
            polarity = 'neutral'

        return {
            'score': sentiment_score,
            'magnitude': sentiment_magnitude,
            'polarity': polarity
        }

    def _detect_emotions(self, text: str) -> List[Dict]:
        """Detect specific emotions in text."""
        emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'glad'],
            'anger': ['angry', 'furious', 'annoyed', 'frustrated'],
            'sadness': ['sad', 'disappointed', 'regret', 'sorry'],
            'urgency': ['urgent', 'asap', 'immediately', 'deadline'],
            'appreciation': ['thank', 'grateful', 'appreciate', 'welcome']
        }

        emotions = defaultdict(int)
        doc = self.nlp(text.lower())

        # Count emotion keywords
        for token in doc:
            for emotion, keywords in emotion_keywords.items():
                if token.text in keywords:
                    emotions[emotion] += 1

        # Convert counts to scores
        total_emotions = sum(emotions.values()) or 1
        return [
            {'emotion': emotion, 'intensity': count / total_emotions}
            for emotion, count in emotions.items()
            if count > 0
        ]

    def _calculate_confidence(self, *sentiments) -> float:
        """Calculate confidence score for sentiment analysis."""
        # Higher confidence if components agree
        scores = [s['score'] for s in sentiments if s['score'] != 0]
        if not scores:
            return 0

        # Calculate standard deviation of scores
        std_dev = np.std(scores)
        # Higher confidence if scores are consistent (low std dev)
        confidence = 1 - min(std_dev, 1)

        return confidence