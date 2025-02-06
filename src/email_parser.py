import email
from email import policy
from typing import Dict, List, Any
import logging
import os
import cv2
from PIL import Image
import pytesseract
import face_recognition
import numpy as np


class EmailParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_email_file(self, email_path: str) -> Dict:
        """Parse a single .eml file and extract basic information."""
        try:
            with open(email_path, 'rb') as f:
                msg = email.message_from_bytes(f.read(), policy=policy.default)
                email_message = email.message_from_binary_file(f)

            email_data = {
                'subject': msg.get('subject', ''),
                'from': msg.get('from', ''),
                'to': msg.get('to', ''),
                'date': msg.get('date', ''),
                'body': self._get_email_body(msg),
                'signature': self._extract_signature(self._get_email_body(msg)),
                'headers': self._extract_header(msg),
                'images': self._extract_embedded_images(email_message)
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
    
    def _extract_embedded_images(self, email_message) -> List[Dict[str, Any]]:
        embedded_images = []
        
        for part in email_message.walk():
            if part.get_content_type().startswith('image/'):
                # Save embedded image
                filename = part.get_filename() or 'embedded_image.jpg'
                with open(filename, 'wb') as f:
                    f.write(part.get_payload(decode=True))
                
                # Analyze image
                analysis = self.analyze_images([filename])
                embedded_images.extend(analysis)
                
                # Clean up temporary file
                os.remove(filename)
        
        return embedded_images
    
    def analyze_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Perform comprehensive image analysis
        """
        analysis_results = []
        
        for image_path in image_paths:
            try:
                # Read image
                img = cv2.imread(image_path)
                pil_img = Image.open(image_path)
                
                # OCR Text Extraction
                ocr_text = pytesseract.image_to_string(pil_img)
                
                # Face Detection
                face_locations = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Logo Detection (basic template matching approach)
                logo_matches = self._detect_logos(img)
                
                analysis_results.append({
                    'image_path': image_path,
                    'ocr_text': ocr_text.strip(),
                    'face_count': len(face_locations),
                    'logo_matches': logo_matches
                })
            
            except Exception as e:
                analysis_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return analysis_results
    
    def _detect_logos(self, image, logo_templates_dir: str = 'logo_templates') -> List[Dict]:
        """
        Simple logo detection using template matching
        """
        logo_matches = []
        
        if not os.path.exists(logo_templates_dir):
            return logo_matches
        
        for template_file in os.listdir(logo_templates_dir):
            template_path = os.path.join(logo_templates_dir, template_file)
            template = cv2.imread(template_path, 0)
            w, h = template.shape[::-1]
            
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            res = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                logo_matches.append({
                    'template': template_file,
                    'location': pt,
                    'size': (w, h)
                })
        
        return logo_matches