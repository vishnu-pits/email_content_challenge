import re
import spacy
from typing import Dict, Optional, Tuple
import logging
from email.utils import parseaddr
import tldextract

class JobCompanyExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common job titles and their variations
        self.job_titles = {
            'executive': [
                'ceo', 'cto', 'cfo', 'coo', 'president', 'vice president', 'vp',
                'director', 'chief', 'head', 'executive'
            ],
            'management': [
                'manager', 'supervisor', 'leader', 'coordinator', 'principal',
                'administrator', 'lead'
            ],
            'engineering': [
                'engineer', 'developer', 'architect', 'programmer', 'analyst',
                'scientist', 'technician', 'specialist'
            ],
            'sales': [
                'sales', 'account executive', 'representative', 'consultant',
                'associate', 'advisor'
            ]
        }

    def extract_job_company(self, email_data: Dict) -> Dict[str, str]:
        """
        Extract job position and company information from email data.
        """
        try:
            # Initialize result
            result = {
                'job_title': None,
                'company': None,
                'department': None,
                'confidence_score': 0.0,
                'source': None
            }

            # Try different extraction methods in order of reliability
            methods = [
                self._extract_from_signature,
                self._extract_from_domain,
                self._extract_from_body
            ]

            for method in methods:
                job_info = method(email_data)
                if job_info['confidence_score'] > result['confidence_score']:
                    result = job_info
                    if result['confidence_score'] > 0.8:  # High confidence threshold
                        break

            return result

        except Exception as e:
            self.logger.error(f"Error extracting job and company info: {str(e)}")
            return {
                'job_title': None,
                'company': None,
                'department': None,
                'confidence_score': 0.0,
                'source': None
            }

    def _extract_from_signature(self, email_data: Dict) -> Dict[str, str]:
        """Extract information from email signature."""
        signature = email_data.get('signature', '')
        if not signature:
            return self._empty_result()

        # Process signature with spaCy
        doc = self.nlp(signature)
        
        # Extract company from organization entities
        companies = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        
        # Look for job title patterns
        job_title = None
        lines = signature.lower().split('\n')
        for line in lines:
            # Check for job title patterns
            for category, titles in self.job_titles.items():
                for title in titles:
                    if title in line:
                        # Extract the full job title using surrounding context
                        job_title = self._extract_full_job_title(line)
                        break
                if job_title:
                    break
        
        # Calculate confidence based on presence of structured information
        confidence = 0.0
        if job_title:
            confidence += 0.5
        if companies:
            confidence += 0.4

        return {
            'job_title': job_title,
            'company': companies[0] if companies else None,
            'department': self._extract_department(signature),
            'confidence_score': confidence,
            'source': 'signature'
        }

    def _extract_from_domain(self, email_data: Dict) -> Dict[str, str]:
        """Extract company information from email domain."""
        from_email = email_data.get('from', '')
        _, email_address = parseaddr(from_email)
        
        if not email_address:
            return self._empty_result()

        # Extract domain
        domain = email_address.split('@')[-1]
        extracted = tldextract.extract(domain)
        company_name = extracted.domain.title()
        
        # Check if it's a common email provider
        common_providers = {'gmail', 'yahoo', 'hotmail', 'outlook', 'aol'}
        if extracted.domain.lower() in common_providers:
            return self._empty_result()

        return {
            'job_title': None,
            'company': company_name,
            'department': None,
            'confidence_score': 0.3,  # Lower confidence for domain-based extraction
            'source': 'email_domain'
        }

    def _extract_from_body(self, email_data: Dict) -> Dict[str, str]:
        """Extract information from email body using NER and pattern matching."""
        body = email_data.get('body', '')
        if not body:
            return self._empty_result()

        doc = self.nlp(body)
        
        # Extract companies
        companies = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        
        # Look for job title patterns
        job_title = None
        for sent in doc.sents:
            sent_text = sent.text.lower()
            # Check for job title indicators
            indicators = ['i am', 'working as', 'my role', 'position of', 'title is']
            if any(indicator in sent_text for indicator in indicators):
                for category, titles in self.job_titles.items():
                    for title in titles:
                        if title in sent_text:
                            job_title = self._extract_full_job_title(sent_text)
                            break
                    if job_title:
                        break

        confidence = 0.0
        if job_title:
            confidence += 0.3
        if companies:
            confidence += 0.2

        return {
            'job_title': job_title,
            'company': companies[0] if companies else None,
            'department': self._extract_department(body),
            'confidence_score': confidence,
            'source': 'email_body'
        }

    def _extract_full_job_title(self, text: str) -> str:
        """Extract complete job title from text context."""
        # Common job title patterns
        patterns = [
            r'(?i)(senior|junior|lead|principal|chief|head|executive)?\s*\w+\s*(manager|engineer|developer|analyst|director|coordinator|specialist)',
            r'(?i)(vp|vice president|director)\s+of\s+\w+',
            r'(?i)(c[A-Za-z]o)'  # Matches CEO, CTO, CFO, etc.
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Clean and normalize the job title
                title = match.group(0).strip()
                return title.title()  # Convert to title case
                
        return None

    def _extract_department(self, text: str) -> Optional[str]:
        """Extract department information from text."""
        # Common department patterns
        dept_patterns = [
            r'(?i)department\s+of\s+(\w+)',
            r'(?i)(\w+)\s+department',
            r'(?i)(\w+)\s+division',
            r'(?i)(\w+)\s+team'
        ]
        
        for pattern in dept_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).title()
                
        return None

    def _empty_result(self) -> Dict[str, str]:
        """Return empty result with zero confidence."""
        return {
            'job_title': None,
            'company': None,
            'department': None,
            'confidence_score': 0.0,
            'source': None
        }