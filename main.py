import streamlit as st
import os
import pandas as pd
import tempfile
import re
import json
from collections import Counter

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
        self.format_topics_for_dataframe = TopicAnalyzer()
        self.job_company_extractor = JobCompanyExtractor()
        self.email_location_extractor = EmailLocationExtractor()

    def process_single_email(self, email_data: dict) -> dict:
        """Process a single email and extract all required information."""
        
        try:
            # Extract contact information
            name = self.contact_extractor.extract_name(email_data)
            result = {
                'Full Name': name,
                'Address': self.contact_extractor.extract_address(email_data.get('signature', '')),
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
                'Phone': self.contact_extractor.extract_phone(email_data.get('signature', '')),
            })

            # Extract email type and basic features
            basic_features = self.basic_extractor.extract_features(email_data)
            result.update(basic_features)

            # Perform sentiment analysis
            result['Sentiment Analysis'] = self.sentiment_analyzer.analyze_sentiment(email_data)
            
            # Extract topics
            topics = self.topic_analyzer.extract_topics(email_data)
            # result['Topics of Interest'] = self.topic_analyzer.format_topics_for_dataframe(topics)
            result['Topics of Interest'] = topics
           
            # Extract Technology Stack
            tech_array = self.topic_analyzer.extract_tech_stack(email_data)
            # result['Technology Stack Used'] = self.topic_analyzer.format_tech_stack_for_dataframe(tech_array)
            result['Technology Stack Used'] = tech_array

            # Image-based Insights
            

            # Identifying networks
            result['Relationship Mapping'] = re.findall(r'[\w\.-]+@[\w\.-]+', email_data.get('to', ''))

            # Email
            from_email = re.search(r'[\w\.-]+@[\w\.-]+', email_data.get('from', ''))
            result['email'] = from_email.group(0) if from_email else ''

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

                # Function to choose the most frequent value or the non-null value if mixed with nulls
                def select_most_frequent(series):
                    return series.mode().iloc[0] if not series.mode().empty else None
                
                # Function to merge unique languages
                def merge_unique_languages(series):
                    unique_languages = set()
                    for langs in series.dropna():
                        unique_languages.update(map(str.strip, langs.split(',')))
                    return ', '.join(sorted(unique_languages)) if unique_languages else None
                
                # Function to consolidate 'Active Email Usage Timeline'
                def consolidate_timeline(series):
                    # Extract dictionaries from the series
                    timelines = [json.loads(json.dumps(item)) for item in series.dropna()]  # Ensure proper dictionary handling
                    
                    # Extract values from dictionaries
                    days = [t['day_of_week'] for t in timelines]
                    hours = [t['hour_of_day'] for t in timelines]
                    business_hours = [t['is_business_hours'] for t in timelines]
                    weekend = [t['is_weekend'] for t in timelines]
                    
                    # Compute the most frequent day of the week
                    most_common_day = Counter(days).most_common(1)[0][0] if days else None
                    # Compute the median hour of activity
                    median_hour = int(pd.Series(hours).median()) if hours else None
                    # Compute boolean majority
                    business_hours_majority = max(set(business_hours), key=business_hours.count) if business_hours else None
                    weekend_majority = max(set(weekend), key=weekend.count) if weekend else None
                    
                    # Consolidated output
                    return {
                        "Most Active Day": most_common_day,
                        "Median Active Hour": median_hour,
                        "Mostly Business Hours": business_hours_majority,
                        "Mostly Weekend Usage": weekend_majority
                    }
                
                # Function to consolidate 'Topics of Interest'
                def consolidate_topics(series):
                    all_topics = [topic for sublist in series.dropna() for topic in sublist]
                    topic_counts = Counter(all_topics)
                    sorted_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)
                    return ', '.join(sorted_topics) if sorted_topics else None
                
                # Function to consolidate 'Technology Stack Discussed'
                def consolidate_tech_stack(series):
                    all_techs = [tech["name"] for sublist in series.dropna() for tech in sublist]
                    tech_counts = Counter(all_techs)
                    sorted_techs = sorted(tech_counts, key=tech_counts.get, reverse=True)
                    return ', '.join(sorted_techs) if sorted_techs else None
                
                # Function to consolidate unique emails in 'Relationship Mapping'
                def consolidate_relationships(series):
                    unique_emails = sorted(set(email for sublist in series.dropna() for email in sublist))
                    return ', '.join(unique_emails) if unique_emails else None
                
                # Sentiment priority mapping
                sentiment_priority = {
                    "very positive": 5,
                    "positive": 4,
                    "neutral": 3,
                    "negative": 2,
                    "very negative": 1
                }

                def consolidate_sentiment(series):
                    sentiment_counts = Counter(series.dropna())
                    most_common = sentiment_counts.most_common()
                    best_sentiment = sorted(most_common, key=lambda x: (-x[1], -sentiment_priority[x[0]]))[0][0]
                    return best_sentiment
                
                
                df_grouped = df.groupby('email', as_index=False).agg({
                    'Full Name': select_most_frequent,
                    'Address': select_most_frequent,
                    'Languages Spoken': merge_unique_languages,
                    'Gender': select_most_frequent,
                    'Job Title': select_most_frequent,
                    'Company': select_most_frequent,
                    'Department': select_most_frequent,
                    'Personal or Businness Email': select_most_frequent,
                    'Country or Region': select_most_frequent,
                    'Active Email Usage Timeline': consolidate_timeline,
                    'Phone': select_most_frequent,
                    'Email Type': select_most_frequent,
                    'Sentiment Analysis': consolidate_sentiment,
                    'Topics of Interest': consolidate_topics,
                    'Technology Stack Used': consolidate_tech_stack,
                    'Relationship Mapping': consolidate_relationships,
                })
                
                # Convert dictionary column to string format for DataFrame display
                df_grouped['Active Email Usage Timeline'] = df_grouped['Active Email Usage Timeline'].apply(json.dumps)


                # Change index to start from 1
                # df.index = df.index + 1
                df_grouped.index = df_grouped.index + 1
                
                # Display results
                st.subheader("Processed Email Analysis")
                st.dataframe(df_grouped)

                # Option to download results
                csv = df_grouped.to_csv(index=False).encode('utf-8')
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