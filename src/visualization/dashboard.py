import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Email Analysis Dashboard", layout="wide")
        self.load_data()

    def load_data(self):
        """Load the processed email data."""
        try:
            self.df = pd.read_csv(Path("data/processed/email_analysis.csv"))
        except FileNotFoundError:
            st.error("No processed data found. Please run the analysis first.")
            self.df = pd.DataFrame()

    def render(self):
        """Render the main dashboard."""
        st.title("Email Analysis Dashboard")

        if self.df.empty:
            return

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", len(self.df))
        with col2:
            if 'from' in self.df:
                st.metric("Unique Senders", self.df['from'].nunique())
            else:
                st.metric("Unique Senders", 0)
        with col3:
            st.metric("Languages Detected", self.df['languages'].nunique())
        with col4:
            st.metric("Average Sentiment", f"{self.df['sentiment']}")

        # Email Type Distribution
        st.subheader("Email Type Distribution")
        fig_types = px.pie(self.df, names='email_type')
        st.plotly_chart(fig_types)

        # Sentiment Analysis Over Time
        st.subheader("Sentiment Analysis Over Time")
        fig_sentiment = px.line(self.df, x='date', y='sentiment',
                              title='Email Sentiment Over Time')
        st.plotly_chart(fig_sentiment)

        # Topic Distribution
        st.subheader("Common Topics")
        topics_data = self.df['topics'].explode()
        fig_topics = px.histogram(topics_data)
        st.plotly_chart(fig_topics)

        # Relationship Network
        st.subheader("Email Relationship Network")
        # Add network visualization here using networkx or similar

        # Raw Data View
        if st.checkbox("Show Raw Data"):
            st.dataframe(self.df)


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.render()