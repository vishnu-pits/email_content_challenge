import email
from email import policy
from typing import Dict, List
import logging

class EmailParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_email_file(self, email_path: str) -> Dict:
        """Parse a single .eml file and extract basic information."""
        try:
            with open(email_path, 'rb') as f:
                msg = email.message_from_bytes(f.read(), policy=policy.default)

            email_data = {
                'subject': msg.get('subject', ''),
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'date': msg.get('date', ''),
                'body': self._get_email_body(msg),
                'signature': self._extract_signature(self._get_email_body(msg)),
                'headers': self._extract_header(msg),
            }
            return email_data
        except Exception as e:
            self.logger.error(f"Error parsing email {email_path}: {str(e)}")
            return {}

    def _get_email_body(self, msg) -> str:
        """Extract the email body from various content types."""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        return msg.get_payload(decode=True).decode()

    def _extract_signature(self, body: str) -> str:
        """Extract email signature using common patterns."""
        signature_markers = [
            '\n--\n',
            '\nRegards',
            '\nBest regards',
            '\nKind regards',
            '\nBest wishes',
            '\nSincerely',
            '\nCheers',
            '\nThanks'
        ]
        
        lowest_idx = len(body)
        for marker in signature_markers:
            idx = body.rfind(marker)
            if idx != -1 and idx < lowest_idx:
                lowest_idx = idx
        
        if lowest_idx < len(body):
            return body[lowest_idx:].strip()
        return ""
    
    def _extract_header(self, msg) -> Dict:
        ip_headers = ['received', 'x-originating-ip', 'x-sender-ip']
        headers = {}

        for header in ip_headers:
            header_value = msg.get(header, '')
            if header_value:
                headers[header] = header_value

        return headers