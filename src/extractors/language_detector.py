from langdetect import detect_langs
from typing import List, Dict
import logging
import spacy
from collections import Counter
from langcodes import Language


class LanguageDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)        

    def detect_languages(self, text: str) -> List[Dict[str, float]]:
        """
        Detect languages in the text and return their names as a comma-separated string.
        """
        try:
            # Use langdetect for initial detection
            lang_probabilities = detect_langs(text)

            # Convert language codes to full language names
            detected_langs = [Language.get(lang.lang).display_name() for lang in lang_probabilities]

            return ", ".join(detected_langs)

        except Exception as e:
            self.logger.error(f"Error detecting language: {str(e)}")
            return "English"