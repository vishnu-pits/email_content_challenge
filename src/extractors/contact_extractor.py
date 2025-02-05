# src/extractors/contact_extractor.py

import re
import spacy
import gender_guesser.detector as gender
import phonenumbers
from typing import Dict, Optional


class ContactExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.gender_detector = gender.Detector()

    def extract_name(self, email_data: Dict) -> str:
        """Extract full name from email data."""
        # Try email signature first
        doc = self.nlp(email_data['signature'])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text

        # Try from email address
        from_email = email_data['from']
        name_match = re.match(r'"?([^"@]+)"?\s*<?[^>]*>', from_email)
        if name_match:
            return name_match.group(1)

        return ""

    def predict_gender(self, name: str) -> str:
        """Predict gender based on first name."""
        first_name = name.split()[0]
        gender_prediction = self.gender_detector.get_gender(first_name)
        return gender_prediction

    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone numbers from text."""
        matches = phonenumbers.PhoneNumberMatcher(text, "US")  # Assumes US format
        for match in matches:
            return phonenumbers.format_number(match.number,
                                              phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return None

    def extract_address(self, text: str) -> Optional[str]:
        """Extract address using NER and pattern matching."""
        doc = self.nlp(text)
        address_parts = []
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                address_parts.append(ent.text)

        # Look for postal codes
        postal_codes = re.findall(r'\b\d{5}(?:-\d{4})?\b', text)
        if postal_codes:
            address_parts.extend(postal_codes)

        return ", ".join(address_parts) if address_parts else None