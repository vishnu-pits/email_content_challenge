from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
import logging
from typing import List, Dict
import numpy as np
from collections import Counter


class TopicAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.95,
            min_df=2
        )
        self.nmf = NMF(
            n_components=10,
            random_state=42
        )

    def extract_topics(self, text: str) -> List[Dict[str, float]]:
        """Extract main topics from the text with their relevance scores."""
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)

            # Extract key phrases and entities
            key_phrases = self._extract_key_phrases(processed_text)
            entities = self._extract_entities(processed_text)

            # Combine and score topics
            topics = self._score_topics(key_phrases, entities)

            return topics

        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")
            return []

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for topic extraction."""
        doc = self.nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct
               and token.is_alpha and len(token.text) > 2
        ]
        return " ".join(tokens)

    def _extract_key_phrases(self, text: str) -> List[Dict[str, float]]:
        """Extract key phrases using TF-IDF and NMF."""
        try:
            # Fit and transform the text
            tfidf_matrix = self.vectorizer.fit_transform([text])

            # Apply NMF
            nmf_output = self.nmf.fit_transform(tfidf_matrix)

            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()

            # Extract top words for each topic
            key_phrases = []
            for topic_idx, topic in enumerate(self.nmf.components_):
                top_words_idx = topic.argsort()[:-10:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                score = float(nmf_output[0][topic_idx])
                if score > 0.1:  # Filter low-relevance topics
                    key_phrases.append({
                        "phrase": " ".join(top_words[:3]),
                        "score": score
                    })

            return key_phrases

        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {str(e)}")
            return []

    def _extract_entities(self, text: str) -> List[Dict[str, float]]:
        """Extract named entities and their frequencies."""
        doc = self.nlp(text)
        entities = []
        entity_counts = Counter()

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'TOPIC']:
                entity_counts[ent.text.lower()] += 1

        # Convert counts to scores
        total = sum(entity_counts.values())
        if total > 0:
            for entity, count in entity_counts.most_common():
                score = count / total
                if score > 0.1:  # Filter low-frequency entities
                    entities.append({
                        "entity": entity,
                        "score": score
                    })

        return entities

    def _score_topics(self, key_phrases: List[Dict], entities: List[Dict]) -> List[Dict]:
        """Combine and score topics from key phrases and entities."""
        topics = []
        seen_topics = set()

        # Combine key phrases and entities, removing duplicates
        for item in key_phrases + entities:
            topic_text = item.get("phrase", item.get("entity"))
            if topic_text and topic_text not in seen_topics:
                topics.append({
                    "topic": topic_text,
                    "score": item["score"],
                    "type": "phrase" if "phrase" in item else "entity"
                })
                seen_topics.add(topic_text)

        # Sort by score
        topics.sort(key=lambda x: x["score"], reverse=True)

        return topics[:10]  # Return top 10 topics