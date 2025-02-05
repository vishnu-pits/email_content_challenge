import re
from typing import Dict
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime


class BasicExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_features(self, email_data: Dict) -> Dict:
        """
        Extract basic features from email data including email type,
        formality level, length metrics, and structural characteristics.
        """
        try:
            body = email_data.get('body', '')
            subject = email_data.get('subject', '')

            features = {
                'email_type': self._determine_email_type(email_data),
                'formality_level': self._assess_formality(body),
                'metrics': self._calculate_metrics(body),
                'has_signature': bool(email_data.get('signature')),
                'has_attachments': self._check_attachments(email_data),
                'urgency_level': self._determine_urgency(subject, body),
                'time_characteristics': self._extract_time_characteristics(email_data)
            }

            return features

        except Exception as e:
            self.logger.error(f"Error extracting basic features: {str(e)}")
            return {}

    def _determine_email_type(self, email_data: Dict) -> str:
        """Determine the type of email based on content and structure."""
        body = email_data.get('body', '').lower()
        subject = email_data.get('subject', '').lower()

        # Check for marketing indicators
        marketing_keywords = ['subscribe', 'offer', 'discount', 'sale', 'newsletter']
        if any(keyword in body or keyword in subject for keyword in marketing_keywords):
            return 'marketing'

        # Check for automated notifications
        if any(x in body for x in ['this is an automated', 'do not reply', 'system notification']):
            return 'automated'

        # Check for formal business communication
        business_indicators = ['meeting', 'proposal', 'contract', 'project', 'report']
        if any(indicator in body or indicator in subject for indicator in business_indicators):
            return 'business'

        # Check for personal communication
        personal_indicators = ['hey', 'hi', 'hello', 'thanks', 'thank you', 'regards']
        if any(indicator in body.lower() for indicator in personal_indicators):
            return 'personal'

        return 'other'

    def _assess_formality(self, text: str) -> str:
        """Assess the formality level of the email text."""
        text = text.lower()

        # Formal indicators
        formal_indicators = {
            'dear': 2,
            'sincerely': 2,
            'regards': 1,
            'best regards': 2,
            'thank you': 1,
            'please': 1,
            'kindly': 2
        }

        # Informal indicators
        informal_indicators = {
            'hey': -2,
            'hi': -1,
            'thanks': -1,
            'cheers': -1,
            'btw': -2,
            '!': -0.5,
            'lol': -2
        }

        formality_score = 0

        # Calculate formal score
        for indicator, weight in formal_indicators.items():
            formality_score += text.count(indicator) * weight

        # Calculate informal score
        for indicator, weight in informal_indicators.items():
            formality_score += text.count(indicator) * weight

        # Determine formality level
        if formality_score > 3:
            return 'very_formal'
        elif formality_score > 1:
            return 'formal'
        elif formality_score > -1:
            return 'neutral'
        elif formality_score > -3:
            return 'informal'
        else:
            return 'very_informal'

    def _calculate_metrics(self, text: str) -> Dict:
        """Calculate various text metrics."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'contains_urls': bool(
                re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
            'contains_numbers': bool(re.search(r'\d+', text))
        }

    def _check_attachments(self, email_data: Dict) -> bool:
        """Check if email has attachments."""
        # Look for common attachment indicators
        attachment_indicators = [
            'Content-Disposition: attachment',
            'Content-Type: application/',
            'Content-Type: image/'
        ]

        raw_email = str(email_data)
        return any(indicator in raw_email for indicator in attachment_indicators)

    def _determine_urgency(self, subject: str, body: str) -> str:
        """Determine the urgency level of the email."""
        text = (subject + ' ' + body).lower()

        # Urgent indicators
        urgent_indicators = [
            'urgent', 'asap', 'emergency', 'immediate attention',
            'deadline', 'important', 'priority', 'time-sensitive'
        ]

        # Count urgent indicators
        urgency_score = sum(text.count(indicator) for indicator in urgent_indicators)

        # Determine urgency level
        if urgency_score > 2:
            return 'high'
        elif urgency_score > 0:
            return 'medium'
        else:
            return 'low'

    def _extract_time_characteristics(self, email_data: Dict) -> Dict:
        """Extract time-related characteristics from the email."""
        try:
            date_str = email_data.get('date')
            if date_str:
                date = parsedate_to_datetime(date_str)
                return {
                    'hour_of_day': date.hour,
                    'day_of_week': date.strftime('%A'),
                    'is_weekend': date.weekday() >= 5,
                    'is_business_hours': 9 <= date.hour <= 17
                }
        except Exception as e:
            self.logger.error(f"Error extracting time characteristics: {str(e)}")

        return {
            'hour_of_day': None,
            'day_of_week': None,
            'is_weekend': None,
            'is_business_hours': None
        }