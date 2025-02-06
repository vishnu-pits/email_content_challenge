from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from typing import List, Dict, Set
from collections import Counter
import re
import nltk


class TopicAnalyzer:
    def __init__(self):
        """
        Initialize topic extraction with necessary NLP resources.
        """

        self.nlp = spacy.load("en_core_web_sm")

        nltk.download('stopwords')
        nltk.download('punkt_tab')

        # Predefined domain-specific topics
        self.domain_topics = {
            'business': [
                'strategy', 'management', 'finance', 'marketing', 
                'sales', 'operations', 'consulting', 'leadership'
            ],
            'technology': [
                'innovation', 'digital transformation', 'ai', 
                'machine learning', 'cloud computing', 'cybersecurity'
            ],
            'science': [
                'research', 'innovation', 'discovery', 'methodology', 
                'experiment', 'analysis', 'data'
            ],
            'hr': [
                'recruitment', 'training', 'development', 'hiring', 
                'performance', 'team building'
            ]
        }

        # Stop words to filter out
        self.stop_words = set(nltk.corpus.stopwords.words('english'))

    def extract_topics(self, email_data: Dict) -> List[str]:
        """Extract topics of interest from email content."""

        # Combine all text sources
        full_text = ' '.join([
            email_data.get('subject', ''),
            email_data.get('body', ''),
            email_data.get('signature', '')
        ]).lower()

        # Clean and preprocess text
        full_text = self._preprocess_text(full_text)

        # Extract topics using multiple methods
        topics = set()

        # 1. Named Entity Recognition
        topics.update(self._extract_named_entities(full_text))

        # 2. TF-IDF based keyword extraction
        topics.update(self._extract_tfidf_keywords(full_text))

        # 3. Domain-specific topic matching
        topics.update(self._match_domain_topics(full_text))

        # 4. Noun phrase extraction
        topics.update(self._extract_noun_phrases(full_text))

        # Filter and clean topics
        filtered_topics = self._filter_topics(topics)

        return list(filtered_topics)

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for topic extraction."""
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Convert to lowercase
        text = text.lower()

        return text

    def _extract_tfidf_keywords(self, text: str, top_n: int = 5) -> Set[str]:
        """Extract key phrases using TF-IDF and NMF."""
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        
        # Remove stop words and short words
        tokens = [
            token.lower() for token in tokens 
            if token.lower() not in self.stop_words 
            and len(token) > 2
        ]
        
        # If not enough tokens, return empty set
        if len(tokens) < top_n:
            return set()
        
        # Create TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names and their TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Sort keywords by TF-IDF score
        keywords = sorted(
            zip(feature_names, tfidf_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return set(keyword for keyword, _ in keywords[:top_n])

    def _extract_named_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            # Focus on specific entity types
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
                entities.add(ent.text.lower())
        
        return entities

    def _match_domain_topics(self, text: str) -> Set[str]:
        """
        Match text against predefined domain topics.
        """
        matched_topics = set()
        
        for domain, topics in self.domain_topics.items():
            for topic in topics:
                if topic.lower() in text:
                    matched_topics.add(topic)
        
        return matched_topics
    
    def _extract_noun_phrases(self, text: str) -> Set[str]:
        """
        Extract meaningful noun phrases.
        """
        doc = self.nlp(text)
        noun_phrases = set()
        
        for chunk in doc.noun_chunks:
            # Filter out very short or generic phrases
            if len(chunk.text.split()) > 1 and len(chunk.text) > 3:
                noun_phrases.add(chunk.text.lower())
        
        return noun_phrases
    
    def _filter_topics(self, topics: Set[str]) -> Set[str]:
        """
        Filter and clean extracted topics.
        """
        filtered = set()
        
        for topic in topics:
            # Remove single characters and very short topics
            if len(topic) > 2:
                # Remove stop words and common generic terms
                words = [
                    word for word in topic.split() 
                    if word not in self.stop_words
                ]
                
                if words:
                    filtered.add(' '.join(words))
        
        return filtered
    
    def format_topics_for_dataframe(self, topics: List[str]) -> str:
        """
        Convert topics list into a formatted string for dataframe.
        """
        if not topics:
            return "No topics detected"
        
        # Sort topics alphabetically
        sorted_topics = sorted(topics)
        
        return '; '.join(sorted_topics)
    

    def extract_tech_stack(self, email_data: Dict) -> List[Dict[str, str]]:
        """
        Extract mentioned technologies and tools from email content.
        Returns a list of dictionaries containing technology name and its category.
        """
        body = email_data.get('body', '').lower()
        subject = email_data.get('subject', '').lower()
        
        # Combine subject and body for analysis
        content = f"{subject}\n{body}"
        
        # Technology categories and their related terms
        tech_patterns = {
            'programming_languages': {
                'pattern': r'\b(python|java|javascript|js|typescript|ts|c\+\+|ruby|php|go|golang|rust|swift|kotlin|scala|r|matlab|perl)\b',
                'exclude': r'(java\.util|\.py\b|\.js\b|\.ts\b|\.rb\b|\.php\b|\.go\b|\.rs\b|\.swift\b|\.kt\b|\.scala\b|\.r\b|\.m\b|\.pl\b)'  # Exclude file extensions
            },
            'frameworks': {
                'pattern': r'\b(django|flask|fastapi|spring|springboot|react|angular|vue|express|node\.?js|rails|laravel|symfony|asp\.net|flutter|pytorch|tensorflow)\b',
                'exclude': None
            },
            'databases': {
                'pattern': r'\b(sql|mysql|postgresql|postgres|oracle|mongodb|mongo|cassandra|redis|elasticsearch|dynamodb|firebase|neo4j|sqlite)\b',
                'exclude': None
            },
            'cloud_services': {
                'pattern': r'\b(aws|amazon|azure|gcp|google cloud|heroku|digitalocean|kubernetes|docker|terraform|jenkins|github actions)\b',
                'exclude': None
            },
            'tools': {
                'pattern': r'\b(git|github|gitlab|bitbucket|jira|confluence|slack|teams|vscode|intellij|pycharm|eclipse|postman|swagger|kubernetes|docker)\b',
                'exclude': None
            },
            'libraries': {
                'pattern': r'\b(numpy|pandas|scipy|scikit-learn|sklearn|requests|beautifulsoup|selenium|junit|pytest|jest|mocha|chai)\b',
                'exclude': None
            },
            'ml_ai': {
                'pattern': r'\b(machine learning|ml|artificial intelligence|ai|deep learning|dl|neural networks|nlp|computer vision|cv|transformers|bert|gpt)\b',
                'exclude': None
            }
        }
        
        def clean_tech_name(tech: str) -> str:
            """Clean and standardize technology names."""
            tech_mappings = {
                'js': 'JavaScript',
                'ts': 'TypeScript',
                'python': 'Python',
                'java': 'Java',
                'nodejs': 'Node.js',
                'node.js': 'Node.js',
                'react': 'React',
                'ml': 'Machine Learning',
                'ai': 'Artificial Intelligence',
                'dl': 'Deep Learning',
                'nlp': 'Natural Language Processing',
                'cv': 'Computer Vision'
            }
            return tech_mappings.get(tech.lower(), tech.title())

        found_technologies = []
        seen_technologies = set()  # To avoid duplicates
        
        # Extract technologies using regex patterns
        for category, patterns in tech_patterns.items():
            matches = re.finditer(patterns['pattern'], content, re.IGNORECASE)
            
            for match in matches:
                tech = match.group(0)
                
                # Skip if it matches exclusion pattern
                if patterns['exclude'] and re.search(patterns['exclude'], tech, re.IGNORECASE):
                    continue
                    
                # Clean and standardize the technology name
                clean_tech = clean_tech_name(tech)
                
                # Avoid duplicates
                if clean_tech.lower() not in seen_technologies:
                    seen_technologies.add(clean_tech.lower())
                    found_technologies.append({
                        'name': clean_tech,
                        'category': category.replace('_', ' ').title()
                    })
        
        # Extract version numbers if available
        for tech in found_technologies:
            version_pattern = fr"\b{tech['name']}\s*(?:version|v)?\.?\s*(\d+(?:\.\d+)*)\b"
            version_match = re.search(version_pattern, content, re.IGNORECASE)
            if version_match:
                tech['version'] = version_match.group(1)
        
        # Sort by category and name
        found_technologies.sort(key=lambda x: (x['category'], x['name']))
        
        return found_technologies

    def get_tech_stack_summary(self, technologies: List[Dict[str, str]]) -> str:
        """
        Generate a human-readable summary of the detected technology stack.
        """
        if not technologies:
            return "No specific technologies mentioned."
        
        # Group technologies by category
        tech_by_category = {}
        for tech in technologies:
            category = tech['category']
            if category not in tech_by_category:
                tech_by_category[category] = []
            
            # Add version if available
            tech_name = tech['name']
            if 'version' in tech:
                tech_name += f" v{tech['version']}"
                
            tech_by_category[category].append(tech_name)
        
        # Build summary
        summary_parts = []
        for category, techs in tech_by_category.items():
            tech_list = ', '.join(techs)
            summary_parts.append(f"{category}: {tech_list}")
        
        return '\n'.join(summary_parts)
    
    def format_tech_stack_for_dataframe(self, technologies: List[Dict[str, str]]) -> str:
        """
        Convert technology stack array into a formatted string for dataframe display.
        Returns a string with technologies grouped by category.
        """
        if not technologies:
            return "No technologies detected"
        
        # Group technologies by category
        tech_by_category = {}
        for tech in technologies:
            category = tech['category']
            if category not in tech_by_category:
                tech_by_category[category] = []
            tech_by_category[category].append(tech['name'])
        
        # Create formatted string
        result_parts = []
        for category, techs in tech_by_category.items():
            tech_list = ', '.join(techs)
            result_parts.append(f"{category}: {tech_list}")
        
        # Join all parts with semicolons for better readability in dataframe
        return '; '.join(result_parts)

    def extract_and_format_tech_stack(self, email_data: Dict) -> str:
        """
        Wrapper function to extract and format technology stack in one go.
        """
        tech_stack = self.extract_tech_stack(email_data)
        return self.format_tech_stack_for_dataframe(tech_stack)