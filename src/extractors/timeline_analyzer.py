from datetime import datetime, timedelta
from typing import List, Dict
import logging
from email.utils import parsedate_to_datetime
import pytz
from collections import defaultdict

class TimelineAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.utc = pytz.UTC

    def analyze_timeline(self, emails: List[Dict]) -> Dict:
        """
        Analyze email usage patterns and timeline.
        Returns comprehensive timeline analysis.
        """
        try:
            # Sort emails by date
            dated_emails = [
                (self._parse_date(email.get('date')), email)
                for email in emails
                if self._parse_date(email.get('date'))
            ]
            dated_emails.sort(key=lambda x: x[0])

            if not dated_emails:
                return self._empty_analysis()

            # Perform various timeline analyses
            return {
                'usage_span': self._analyze_usage_span(dated_emails),
                'activity_patterns': self._analyze_activity_patterns(dated_emails),
                'frequency_analysis': self._analyze_frequency(dated_emails),
                'timeline_segments': self._create_timeline_segments(dated_emails),
                'gaps_analysis': self._analyze_gaps(dated_emails)
            }

        except Exception as e:
            self.logger.error(f"Error in timeline analysis: {str(e)}")
            return self._empty_analysis()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            return parsedate_to_datetime(date_str).replace(tzinfo=self.utc)
        except Exception:
            return None

    def _analyze_usage_span(self, dated_emails: List[tuple]) -> Dict:
        """Analyze the total span of email usage."""
        start_date = dated_emails[0][0]
        end_date = dated_emails[-1][0]
        duration = end_date - start_date

        return {
            'first_email_date': start_date.isoformat(),
            'last_email_date': end_date.isoformat(),
            'total_days': duration.days,
            'active_months': (duration.days // 30) + 1
        }

    def _analyze_activity_patterns(self, dated_emails: List[tuple]) -> Dict:
        """Analyze patterns in email activity."""
        patterns = {
            'hourly': defaultdict(int),
            'daily': defaultdict(int),
            'monthly': defaultdict(int)
        }

        for date, _ in dated_emails:
            patterns['hourly'][date.hour] += 1
            patterns['daily'][date.strftime('%A')] += 1
            patterns['monthly'][date.strftime('%B')] += 1

        # Find peak activity times
        peak_times = {
            'hour': max(patterns['hourly'].items(), key=lambda x: x[1])[0],
            'day': max(patterns['daily'].items(), key=lambda x: x[1])[0],
            'month': max(patterns['monthly'].items