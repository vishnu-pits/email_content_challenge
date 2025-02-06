import spacy
import re
import socket
import whois
import requests
import ipaddress
import tldextract
from email import parser, policy
from typing import Dict, Optional

class EmailLocationExtractor:
    def __init__(self):
        """
        Initialize the extractor with required models
        """
        # Load SpaCy model for NER
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common country TLDs
        self.country_tlds = {
            'uk': 'United Kingdom',
            'de': 'Germany',
            'fr': 'France',
            'au': 'Australia',
            'ca': 'Canada',
            'cn': 'China',
            'in': 'India',
            'it': 'Italy',
            'es': 'Spain',
            'us': 'United States',
            'jp': 'Japan',
            'se': 'Sweeden',
            'be': 'Belgium',
            'pl': 'Poland',
            'ch': 'Switzerland',
            'nl': 'Netherlands'
        }

    def extract_from_signature(self, signature: str) -> Optional[str]:
        """Extract country from email signature using NER."""
        if not signature:
            return None
        
        doc = self.nlp(signature)
        
        # Look for country or GPE (geo-political entity) entities
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                return ent.text
        return None

    def extract_from_domain(self, email: str) -> Optional[str]:
        """Extract country information from email domain."""
        if not email:
            return None
        
        try:
            domain = email.split('@')[1].strip('<>')
        except IndexError:
            return None

        # Extract domain parts
        ext = tldextract.extract(domain)
        
        # Check if it's a country TLD
        if ext.suffix in self.country_tlds:
            return self.country_tlds[ext.suffix]
        
        if ext.suffix not in ['com', 'org', 'net', 'edu', 'gov', 'mil']:
            try:
                # Try WHOIS lookup
                w = whois.whois(f"{ext.domain}.{ext.suffix}")
                if w.country:
                    return w.country
            except Exception:
                pass
        
        return None
    
    def _get_ip_location(self, ip: str) -> Optional[str]:
        """Get location from IP using free IP-API."""
        try:
            if ipaddress.ip_address(ip).is_private:
                return None
                
            response = requests.get(f'http://ip-api.com/json/{ip}')
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('country')
        except Exception:
            pass
        return None

    def extract_from_ip(self, email_headers: Dict) -> Optional[str]:
        """Extract country from email headers containing IP addresses."""
        # Common header fields that might contain IPs
        ip_headers = ['Received', 'X-Originating-IP', 'X-Sender-IP']
        
        for header in ip_headers:
            if header in email_headers:
                # Extract IP addresses using regex
                ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                ips = re.findall(ip_pattern, email_headers[header])
                
                for ip in ips:
                    # Look up IP location
                    location = self._get_ip_location(ip)
                    if location:
                        return location
        return None

    def extract_location(self, email_data: Dict) -> str:
        
        location = self.extract_from_signature(email_data['signature'])
        if location:
            return location
        
        location = self.extract_from_domain(email_data['from'])
        if location:
            return location
        
        location = self.extract_from_ip(email_data['headers'])
        if location:
            return location

        return None

    def __del__(self):
        """Clean up GeoIP reader."""
        self.geo_reader.close()