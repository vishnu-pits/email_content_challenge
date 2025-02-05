from langdetect import detect_langs
from typing import List, Dict
import logging
import spacy
from collections import Counter


class LanguageDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            # Load spaCy models for multiple languages
            self.nlp_en = spacy.load("en_core_web_sm")
            self.nlp_es = spacy.load("es_core_news_sm")
            self.nlp_fr = spacy.load("fr_core_news_sm")
        except Exception as e:
            self.logger.error(f"Error loading spaCy models: {str(e)}")

    def detect_languages(self, text: str) -> List[Dict[str, float]]:
        """
        Detect languages in the text with confidence scores.
        Returns list of language codes with confidence scores.
        """
        try:
            # Use langdetect for initial detection
            lang_probabilities = detect_langs(text)

            # Convert to list of dictionaries
            detected_langs = [
                {"lang": lang.lang, "confidence": lang.prob}
                for lang in lang_probabilities
            ]

            # Additional validation using spaCy
            main_lang = detected_langs[0]["lang"] if detected_langs else "en"
            if main_lang == "en" and self._validate_english(text):
                detected_langs[0]["confidence"] += 0.1

            return detected_langs

        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return [{"lang": "en", "confidence": 1.0}]  # Default to English

    def _validate_english(self, text: str) -> bool:
        """Validate if text is truly English using spaCy."""
        try:
            doc = self.nlp_en(text[:1000])  # Limit text length for performance
            # Check if text contains reasonable English words
            return len([token for token in doc if token.is_alpha]) > 0
        except Exception:
            return True

    def detect_multilingual_segments(self, text: str) -> Dict[str, List[str]]:
        """Identify different language segments within the text."""
        segments = {}
        current_lang = None
        current_segment = []

        for sentence in text.split('.'):
            try:
                lang = detect_langs(sentence)[0].lang
                if lang != current_lang:
                    if current_lang:
                        if current_lang not in segments:
                            segments[current_lang] = []
                        segments[current_lang].extend(current_segment)
                    current_segment = []
                    current_lang = lang
                current_segment.append(sentence)
            except:
                continue

        # Add final segment
        if current_lang and current_segment:
            if current_lang not in segments:
                segments[current_lang] = []
            segments[current_lang].extend(current_segment)

        return segments