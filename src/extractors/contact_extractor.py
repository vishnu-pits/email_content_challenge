# src/extractors/contact_extractor.py

import re
import spacy
import gender_guesser.detector as gender
import phonenumbers
from typing import Dict, Optional
import tldextract


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
        name_match = re.match(r'(?:"?([^"<]+)"?\s*)?<[^>]+>', from_email)
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
        matches = phonenumbers.PhoneNumberMatcher(text, None)
        for match in matches:
            # Check if the extracted number is valid
            if phonenumbers.is_valid_number(match.number):
                return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        return None

    def extract_address(self, text: str) -> Optional[str]:
        # Regex for capturing structured addresses
        address_pattern = re.compile(
            r'\b(\d+\s+[A-Za-z0-9\s,|-]+(?:Suite|Ste|Apt|Floor|Building|Block)?\s*\d*,?\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)\b'
        )

        address_match = address_pattern.search(text)
        if address_match:
            # Check if the address is near relevant keywords
            context_window = text[max(0, address_match.start() - 50): address_match.end() + 50]
            if any(keyword in context_window.lower() for keyword in ["address", "located at", "sender", "from"]):
                return address_match.group(0).strip()

        # If regex fails, use NLP-based extraction
        doc = self.nlp(text)
        address_parts = []
        postal_codes = []

        for ent in doc.ents:
            # Skip entities that are likely not part of an address
            if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY"]:
                continue

            if ent.label_ in ["GPE", "LOC", "FAC", "ORG", "CARDINAL"]:
                address_parts.append(ent.text)

            if ent.label_ == "CARDINAL" and re.match(r"\b\d{5,6}(?:-\d{4})?\b", ent.text):
                postal_codes.append(ent.text)

        if postal_codes:
            address_parts.extend(postal_codes)

        return ", ".join(address_parts) if address_parts else None
    

    def classify_email(self, email_input: str) -> str:
        personal_domains = {
            "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com",
            "icloud.com", "protonmail.com", "zoho.com", "yandex.com", "mail.com"
        }

        # Extract email address using regex
        match = re.search(r'[\w\.-]+@[\w\.-]+', email_input)
        if not match:
            raise ValueError("Invalid email address: no email pattern found.")

        email = match.group()
        domain = email.split("@")[-1].lower()
        extracted = tldextract.extract(domain)
        root_domain = f"{extracted.domain}.{extracted.suffix}"

        if root_domain in personal_domains:
            return "Personal"
        else:
            return "Business"