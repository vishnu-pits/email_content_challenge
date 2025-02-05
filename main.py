import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import sys
from typing import Dict, List
import streamlit as st

# Import our custom modules
from src.email_parser import EmailParser
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.basic_extractor import BasicExtractor
from src.extractors.language_detector import LanguageDetector
from src.extractors.sentiment_analyzer import SentimentAnalyzer
from src.extractors.topic_analyzer import TopicAnalyzer


class EmailAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.setup_logging()
        self.load_config(config_path)
        self.initialize_extractors()
        self.create_directories()

    def setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('email_analyzer.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)

    def initialize_extractors(self):
        """Initialize all extractor components."""
        try:
            self.email_parser = EmailParser()
            self.contact_extractor = ContactExtractor()
            self.basic_extractor = BasicExtractor()
            self.language_detector = LanguageDetector()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.topic_analyzer = TopicAnalyzer()
            self.logger.info("All extractors initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing extractors: {str(e)}")
            sys.exit(1)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = ['data/raw', 'data/processed', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def process_single_email(self, email_data: Dict) -> Dict:
        """Process a single email and extract all required information."""
        try:
            # Extract basic information
            result = {
                'timestamp': datetime.now().isoformat(),
                'email_id': email_data.get('message-id', ''),
                'subject': email_data.get('subject', ''),
                'date': email_data.get('date', ''),
                'from': email_data.get('from', ''),
            }

            # Extract contact information
            name = self.contact_extractor.extract_name(email_data)
            result.update({
                'full_name': name,
                'gender': self.contact_extractor.predict_gender(name),
                'phone': self.contact_extractor.extract_phone(email_data.get('body', '')),
                'address': self.contact_extractor.extract_address(email_data.get('body', '')),
            })

            # Extract email type and basic features
            basic_features = self.basic_extractor.extract_features(email_data)
            result.update(basic_features)

            # Detect languages
            result['languages'] = self.language_detector.detect_languages(email_data.get('body', ''))

            # Perform sentiment analysis
            result['sentiment'] = self.sentiment_analyzer.analyze_sentiment(email_data.get('body', ''))

            # Extract topics
            result['topics'] = self.topic_analyzer.extract_topics(email_data.get('body', ''))

            return result

        except Exception as e:
            self.logger.error(f"Error processing email: {str(e)}")
            return {}

    def process_emails(self, input_directory: str) -> pd.DataFrame:
        """Process all emails in the input directory."""
        self.logger.info(f"Starting email processing from directory: {input_directory}")

        # Parse all emails
        email_data_list = self.email_parser.process_directory(input_directory)
        self.logger.info(f"Found {len(email_data_list)} emails to process")

        # Process each email
        results = []
        for idx, email_data in enumerate(email_data_list, 1):
            self.logger.info(f"Processing email {idx}/{len(email_data_list)}")
            processed_data = self.process_single_email(email_data)
            if processed_data:
                results.append(processed_data)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df

    def save_results(self, df: pd.DataFrame, output_path: str):
        """Save the results to a CSV file."""
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Results saved successfully to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

    def launch_dashboard(self):
        """Launch the Streamlit dashboard."""
        try:
            subprocess.Popen([
                "streamlit", "run",
                "src/visualization/dashboard.py"
            ])
            self.logger.info("Dashboard launched successfully")
        except Exception as e:
            self.logger.error(f"Error launching dashboard: {str(e)}")


def main():
    # Initialize the analyzer
    analyzer = EmailAnalyzer()

    # Process emails
    input_dir = analyzer.config.get('input_directory', 'data/raw')
    output_file = analyzer.config.get('output_file', 'data/processed/email_analysis.csv')

    df = analyzer.process_emails(input_dir)

    # Save results
    analyzer.save_results(df, output_file)

    # Launch dashboard
    analyzer.launch_dashboard()


if __name__ == "__main__":
    main()