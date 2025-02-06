import streamlit as st
import os
import pandas as pd
import tempfile

# Import our custom modules
from src.email_parser import EmailParser
from src.extractors.contact_extractor import ContactExtractor
from src.extractors.basic_extractor import BasicExtractor
from src.extractors.language_detector import LanguageDetector
from src.extractors.sentiment_analyzer import SentimentAnalyzer
from src.extractors.topic_analyzer import TopicAnalyzer
from src.extractors.job_company_extractor import JobCompanyExtractor
from src.extractors.email_location_extractor import EmailLocationExtractor

class EmailAnalyzer:
    def __init__(self):
        # Initialize extractors
        self.email_parser = EmailParser()
        self.contact_extractor = ContactExtractor()
        self.basic_extractor = BasicExtractor()
        self.language_detector = LanguageDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_analyzer = TopicAnalyzer()
        self.job_company_extractor = JobCompanyExtractor()
        self.email_location_extractor = EmailLocationExtractor()

    def process_single_email(self, email_data: dict) -> dict:
        """Process a single email and extract all required information."""
        try:
            # Extract contact information
            name = self.contact_extractor.extract_name(email_data)
            result = {
                'Full Name': name,
                'Address': self.contact_extractor.extract_address(email_data.get('body', '')),
                'Languages Spoken': self.language_detector.detect_languages(email_data.get('body', '')),
                'Gender': self.contact_extractor.predict_gender(name),
            }

            # Add job and company extraction
            job_company_info = self.job_company_extractor.extract_job_company(email_data)
            result.update({
                'Job Title': job_company_info['job_title'],
                'Company': job_company_info['company'],
                'Department': job_company_info['department'],
            })

            # Extract basic information
            result.update({
                'Personal or Businness Email': self.contact_extractor.classify_email(email_data.get('from', '')),
                'Country or Region': self.email_location_extractor.extract_location(email_data),
                'phone': self.contact_extractor.extract_phone(email_data.get('body', '')),
            })

            # Extract email type and basic features
            basic_features = self.basic_extractor.extract_features(email_data)
            result.update(basic_features)

            result.update({
                'email_id': email_data.get('message-id', ''),
                'subject': email_data.get('subject', ''),
                'date': email_data.get('date', ''),
            })

            # Perform sentiment analysis
            sentiment = self.sentiment_analyzer.analyze_sentiment(email_data)
            result['sentiment'] = sentiment.get('overall_score', 0)

            # Extract topics
            result['topics'] = self.topic_analyzer.extract_topics(email_data.get('body', ''))

            return result

        except Exception as e:
            st.error(f"Error processing email: {str(e)}")
            return {}

def main():
    st.title("Email Analysis Dashboard")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload .eml files", 
        type=['eml'], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create a temporary directory to save uploaded files
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(tmpdirname, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(file_path)

            # Process emails
            analyzer = EmailAnalyzer()
            
            # Parse emails
            st.write(f"Processing {len(file_paths)} email files...")
            email_data_list = []
            for file_path in file_paths:
                try:
                    parsed_email = analyzer.email_parser.parse_email_file(file_path)
                    email_data_list.append(parsed_email)
                except Exception as e:
                    st.error(f"Error parsing {file_path}: {str(e)}")

            # Process each email
            results = []
            progress_bar = st.progress(0)
            for i, email_data in enumerate(email_data_list):
                processed_data = analyzer.process_single_email(email_data)
                if processed_data:
                    results.append(processed_data)
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(email_data_list))

            # Convert to DataFrame
            if results:
                df = pd.DataFrame(results)

                # Change index to start from 1
                df.index = df.index + 1
                
                # Display results
                st.subheader("Processed Email Analysis")
                st.dataframe(df)

                # Option to download results
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='email_analysis_results.csv',
                    mime='text/csv',
                )

            else:
                st.warning("No emails could be processed.")

if __name__ == "__main__":
    main()