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
                'Active Email Usage Timeline': self._extract_time_characteristics(email_data),
                'Email Type': self._determine_email_type(email_data),
            }

            return features

        except Exception as e:
            self.logger.error(f"Error extracting basic features: {str(e)}")
            return {}

    def _determine_email_type(self, email_data: Dict) -> str:
        """
        Determine the type of email using a scoring system based on multiple indicators.
        
        Types:
        - formal: Professional business communication
        - casual: Informal business or peer communication
        - transactional: Task-oriented, action-required emails
        - marketing: Promotional content
        - automated: System-generated notifications
        """
        body = email_data.get('body', '').lower()
        subject = email_data.get('subject', '').lower()
        signature = email_data.get('signature', '').lower()
        
        # Initialize scores
        scores = {
            'formal': 0,
            'casual': 0,
            'transactional': 0,
            'marketing': 0,
            'automated': 0
        }
        
        # Marketing indicators
        marketing_patterns = [
            (r'\b(subscribe|unsubscribe|offer|discount|sale|off|save|deal|promotion)\b', 2),
            (r'\b(newsletter|limited time|exclusive|special)\b', 1),
            (r'%(org)s newsletter', 2),
            (r'\b(buy|shop|order now)\b', 1)
        ]
        
        # Automated notification indicators
        automated_patterns = [
            (r'\b(this is an automated|do not reply|system notification|automatic)\b', 3),
            (r'\b(generated|notification|alert|system|automated)\b', 1),
            (r'@no-?reply', 3),
            (r'\b(ticket|case|incident) #\d+', 2)
        ]
        
        # Formal business indicators
        formal_patterns = [
            (r'\b(dear|to whom it may concern|sincerely|yours faithfully)\b', 2),
            (r'\b(meeting|proposal|contract|report|agenda|board|client)\b', 1),
            (r'\b(please find attached|as discussed|regarding|with reference to)\b', 2),
            (r'\b(appreciate your consideration|look forward to|professional)\b', 1),
            (r'[A-Z][a-z]+ [A-Z][a-z]+\s*\n.*\n.*Manager|Director|CEO', 2),  # Professional signature
            (r'\b(confidential|proprietary|business)\b', 1)
        ]
        
        # Casual business indicators
        casual_patterns = [
            (r'\b(hey|hi|hello|hey there)\b(?!.*regards)', 1),  # Greetings not part of signature
            (r'\b(thanks|cheers|talk soon|catch up)\b', 1),
            (r'^\s*hi\s+team', 1),
            (r'\b(quick|heads up|fyi|question)\b', 1),
            (r'[!]{2,}|\?{2,}', 1),  # Multiple punctuation
            (r'\b(great|awesome|cool)\b', 1)
        ]
        
        # Transactional indicators
        transactional_patterns = [
            (r'\b(action required|please review|deadline|due date|reminder)\b', 2),
            (r'\b(approve|reject|confirm|verify|validate|complete)\b', 1),
            (r'\b(form|document|submission|application|request)\b', 1),
            (r'\b(status|update|processed|completed|pending)\b', 1),
            (r'by (today|tomorrow|\d{1,2}/\d{1,2})', 2),
            (r'\b(password|account|login|access)\b', 1)
        ]
        
        def apply_patterns(text: str, patterns: list, category: str):
            for pattern, weight in patterns:
                if re.search(pattern, text, re.I):
                    scores[category] += weight
        
        # Combine subject and body for analysis
        full_text = f"{subject}\n{body}"
        
        # Apply all patterns
        apply_patterns(full_text, marketing_patterns, 'marketing')
        apply_patterns(full_text, automated_patterns, 'automated')
        apply_patterns(full_text, formal_patterns, 'formal')
        apply_patterns(full_text, casual_patterns, 'casual')
        apply_patterns(full_text, transactional_patterns, 'transactional')
        
        # Additional context-based rules
        
        # Check for bulk mail headers
        if any(header in email_data.get('headers', {}) for header in ['List-Unsubscribe', 'Precedence: bulk']):
            scores['marketing'] += 3
        
        # Check for automated sender domains
        from_address = email_data.get('from', '').lower()
        if any(x in from_address for x in ['noreply', 'no-reply', 'donotreply', 'system', 'notification']):
            scores['automated'] += 3
        
        # Check for formal signature block
        if signature:
            signature_lines = signature.count('\n')
            if signature_lines >= 4:  # Complex signatures suggest formal emails
                scores['formal'] += 1
            if re.search(r'(title|position):', signature, re.I):
                scores['formal'] += 1
        
        # Check for urgent/important markers
        if re.search(r'\b(urgent|important|asap|priority)\b', subject, re.I):
            scores['transactional'] += 2
        
        # Length-based adjustments
        word_count = len(body.split())
        if word_count < 30:  # Short emails tend to be casual or transactional
            scores['casual'] += 1
        elif word_count > 200:  # Longer emails tend to be formal
            scores['formal'] += 1
        
        # Get the highest scoring category
        max_score = max(scores.values())
        top_categories = [cat for cat, score in scores.items() if score == max_score]
        
        # If there's a tie between casual and formal, prefer casual
        if len(top_categories) > 1 and 'casual' in top_categories and 'formal' in top_categories:
            return 'casual'
        
        # If there's a tie between transactional and automated, prefer automated
        if len(top_categories) > 1 and 'transactional' in top_categories and 'automated' in top_categories:
            return 'automated'
        
        return top_categories[0]

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