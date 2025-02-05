# src/email_parser.py

import email
from email import policy
from pathlib import Path
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
                'signature': self._extract_signature(self._get_email_body(msg))
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
        signature_markers = ['Best regards', 'Regards', 'Sincerely', 'Thanks']
        lines = body.split('\n')
        for i, line in enumerate(lines):
            if any(marker in line for marker in signature_markers):
                return '\n'.join(lines[i:])
        return ''

    def process_directory(self, directory: str) -> List[Dict]:
        """Process all .eml files in a directory."""
        email_data = []
        directory_path = Path(directory)
        for email_file in directory_path.glob('*.eml'):
            parsed_data = self.parse_email_file(str(email_file))
            if parsed_data:
                email_data.append(parsed_data)
        return email_data