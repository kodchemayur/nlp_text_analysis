"""
TEXT PREPROCESSING MODULE
Part 1 of Assignment: Lowercasing, Tokenization, Stopword removal, Lemmatization
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from typing import List, Tuple, Dict
import logging

class TextPreprocessor:
    def __init__(self, config: Dict):
        # Download required NLTK data
        self._download_nltk_resources()
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Configuration
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Add custom stopwords for newsgroups
        self.custom_stopwords = {
            'would', 'could', 'should', 'said', 'also', 'one', 'two', 
            'three', 'first', 'second', 'third', 'year', 'years', 'time',
            'article', 'writes', 'wrote', 'com', 'edu', 'org', 'gov',
            'subject', 'lines', 'organization', 'distribution', 'university'
        }
        self.stop_words.update(self.custom_stopwords)
    
    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def load_20newsgroups_full(self, categories=None, subset='all'):
        """
        Load FULL 20 Newsgroups dataset as per assignment requirements
        Returns: documents, category_names, target_labels
        """
        self.logger.info("Loading 20 Newsgroups dataset (FULL)...")
        
        # Fetch dataset - using ALL data as required
        newsgroups = fetch_20newsgroups(
            subset=subset,  # 'all' = 18846 documents
            categories=categories,
            remove=('headers', 'footers', 'quotes'),  # Standard cleaning
            shuffle=True,
            random_state=42,
            download_if_missing=True
        )
        
        documents = newsgroups.data
        category_names = newsgroups.target_names
        target_labels = newsgroups.target
        
        self.logger.info(f"✓ Loaded {len(documents)} documents (FULL 20 Newsgroups)")
        self.logger.info(f"✓ Categories: {len(category_names)} categories")
        
        # Show document count per category
        from collections import Counter
        cat_counts = Counter(target_labels)
        for idx, (cat_id, count) in enumerate(cat_counts.most_common()):
            self.logger.info(f"  {category_names[cat_id]:25s}: {count:4d} documents")
        
        return documents, category_names, target_labels
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove numbers if configured
        if self.config.get('remove_numbers', True):
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if configured
        if self.config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text: str) -> Tuple[str, List[str]]:
        """Complete preprocessing pipeline for a single document"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            return "", []
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Apply lemmatization if configured
        if self.config.get('lemmatize', True):
            tokens = self.lemmatize_tokens(tokens)
        
        # Filter by minimum token length
        min_length = self.config.get('min_token_length', 2)
        tokens = [token for token in tokens if len(token) >= min_length]
        
        # Reconstruct cleaned text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text, tokens
    
    def preprocess_corpus(self, documents: List[str]) -> Tuple[List[str], List[List[str]]]:
        """
        Preprocess entire corpus
        Returns: cleaned_documents, tokens_list
        """
        self.logger.info("Starting corpus preprocessing...")
        
        # Limit documents if specified for faster processing
        max_docs = self.config.get('max_documents', 2000)
        if len(documents) > max_docs:
            self.logger.info(f"Processing first {max_docs} documents for speed")
            self.logger.info(f"(Set max_documents higher in config.yaml for full dataset)")
            documents = documents[:max_docs]
        
        cleaned_docs = []
        tokens_list = []
        
        for i, doc in enumerate(documents):
            if i % 500 == 0 and i > 0:
                self.logger.info(f"  Processed {i}/{len(documents)} documents")
            
            cleaned_doc, tokens = self.preprocess_text(doc)
            if cleaned_doc:  # Only add non-empty documents
                cleaned_docs.append(cleaned_doc)
                tokens_list.append(tokens)
        
        # Calculate statistics
        total_tokens = sum(len(tokens) for tokens in tokens_list)
        avg_tokens = total_tokens / len(cleaned_docs) if cleaned_docs else 0
        all_tokens = [token for tokens in tokens_list for token in tokens]
        unique_tokens = len(set(all_tokens))
        
        self.logger.info(f"✓ Preprocessing completed:")
        self.logger.info(f"  Documents processed: {len(cleaned_docs):,}")
        self.logger.info(f"  Total tokens: {total_tokens:,}")
        self.logger.info(f"  Average tokens per document: {avg_tokens:.1f}")
        self.logger.info(f"  Unique tokens: {unique_tokens:,}")
        
        return cleaned_docs, tokens_list