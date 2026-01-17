"""
POS TAGGING MODULE
Part 3 of Assignment: POS tagging, distribution, patterns, category comparison
"""
import spacy
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import logging
import os
import numpy as np
import nltk
from nltk import pos_tag as nltk_pos_tag

class POSAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.patterns_to_extract = config.get('patterns_to_extract', ['ADJ+NOUN', 'NOUN+VERB'])
        self.top_n_words = config.get('top_n_words', 10)
        self.compare_categories = config.get('compare_categories', [])
        self.logger = logging.getLogger(__name__)
        
        # Try to load spaCy, fall back to NLTK
        self.use_spacy = False
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.use_spacy = True
            self.logger.info("Using spaCy for POS tagging")
        except:
            self.logger.warning("spaCy not available, using NLTK for POS tagging")
            # Download NLTK POS tagger if needed
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except:
                nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def tag_document_spacy(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging using spaCy"""
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return pos_tags
    
    def tag_document_nltk(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging using NLTK"""
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        # Convert NLTK tags to universal tags
        pos_tags = [(word, self._convert_nltk_tag(tag)) for word, tag in pos_tags]
        return pos_tags
    
    def _convert_nltk_tag(self, nltk_tag: str) -> str:
        """Convert NLTK POS tags to universal tags"""
        # Simplified mapping
        tag_map = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'NOUN', 'NNPS': 'NOUN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 
            'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
            'PRP': 'PRON', 'PRP$': 'PRON',
            'DT': 'DET', 'IN': 'ADP', 'CC': 'CONJ', 'CD': 'NUM'
        }
        return tag_map.get(nltk_tag[:2], 'X')  # 'X' for unknown
    
    def tag_document(self, text: str) -> List[Tuple[str, str]]:
        """POS tag a document using available method"""
        if self.use_spacy:
            return self.tag_document_spacy(text)
        else:
            return self.tag_document_nltk(text)
    
    def tag_corpus(self, documents: List[str], sample_size: int = 500) -> List[List[Tuple[str, str]]]:
        """POS tag entire corpus (sampled for speed)"""
        self.logger.info(f"POS tagging {min(sample_size, len(documents))} documents...")
        
        tagged_corpus = []
        for i, doc in enumerate(documents[:sample_size]):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"  Tagged {i}/{min(sample_size, len(documents))} documents")
            
            pos_tags = self.tag_document(doc)
            tagged_corpus.append(pos_tags)
        
        return tagged_corpus
    
    def compute_pos_distribution(self, tagged_corpus: List[List[Tuple[str, str]]]) -> Dict:
        """Compute distribution of POS tags"""
        pos_counts = Counter()
        word_pos_counts = defaultdict(Counter)
        
        for doc_tags in tagged_corpus:
            for word, pos in doc_tags:
                pos_counts[pos] += 1
                if pos in ['NOUN', 'VERB', 'ADJ']:  # Only track these for top words
                    word_pos_counts[pos][word.lower()] += 1
        
        # Get top words for each major POS category
        top_words = {}
        for pos in ['NOUN', 'VERB', 'ADJ']:
            if pos in word_pos_counts:
                top_words[pos] = word_pos_counts[pos].most_common(self.top_n_words)
        
        return {
            'pos_distribution': dict(pos_counts),
            'top_words_by_pos': top_words
        }
    
    def extract_pos_patterns(self, tagged_corpus: List[List[Tuple[str, str]]]) -> Dict[str, List]:
        """Extract specified POS patterns from corpus"""
        patterns_found = {pattern: [] for pattern in self.patterns_to_extract}
        
        for doc_tags in tagged_corpus:
            for i in range(len(doc_tags) - 1):
                word1, pos1 = doc_tags[i]
                word2, pos2 = doc_tags[i + 1]
                
                # Check for patterns
                if f"{pos1}+{pos2}" in self.patterns_to_extract:
                    patterns_found[f"{pos1}+{pos2}"].append(f"{word1} {word2}")
        
        # Count frequencies for each pattern
        pattern_counts = {}
        for pattern, instances in patterns_found.items():
            if instances:
                pattern_counts[pattern] = Counter(instances).most_common(20)
        
        return pattern_counts
    
    def compare_categories_pos(self, documents: List[str], category_names: List[str], 
                          target_labels: List[int]) -> Dict:
        """Compare POS distributions across two categories as required"""
        if not self.compare_categories or len(self.compare_categories) < 2:
            return {}
        
        comparison_results = {}
        
        for category in self.compare_categories:
            if category in category_names:
                # Get category index
                cat_idx = category_names.index(category)
                self.logger.info(f"Analyzing category: {category} (index: {cat_idx})")
                
                # Get indices of documents belonging to this category
                # Use the original target_labels to find which documents belong to this category
                doc_indices = [i for i, t in enumerate(target_labels) if t == cat_idx]
                
                if not doc_indices:
                    self.logger.warning(f"No documents found for category: {category}")
                    continue
                
                # Limit to reasonable number for performance
                sample_size = min(100, len(doc_indices), len(documents))
                sampled_indices = doc_indices[:sample_size]
                
                # Ensure indices are within bounds of documents list
                valid_indices = [i for i in sampled_indices if i < len(documents)]
                
                if not valid_indices:
                    self.logger.warning(f"No valid document indices for category: {category}")
                    continue
                
                # Get documents for this category
                cat_docs = [documents[i] for i in valid_indices]
                
                # Log info
                self.logger.info(f"  Processing {len(cat_docs)} documents for {category}")
                
                # Tag and analyze
                tagged_corpus = self.tag_corpus(cat_docs, sample_size=min(100, len(cat_docs)))
                
                if not tagged_corpus:
                    self.logger.warning(f"No valid POS tags for category: {category}")
                    continue
                
                pos_dist = self.compute_pos_distribution(tagged_corpus)
                patterns = self.extract_pos_patterns(tagged_corpus)
                
                comparison_results[category] = {
                    'pos_distribution': pos_dist['pos_distribution'],
                    'top_words_by_pos': pos_dist['top_words_by_pos'],
                    'extracted_patterns': patterns,
                    'num_documents_analyzed': len(cat_docs)
                }
            else:
                self.logger.warning(f"Category '{category}' not found in dataset")
                self.logger.info(f"Available categories: {category_names}")
        
        return comparison_results


    def analyze(self, documents: List[str], category_names: List[str], 
                target_labels: List[int]) -> Dict:
        """Complete POS analysis pipeline"""
        self.logger.info("\nStarting POS analysis...")
        
        # 1. Tag corpus
        tagged_corpus = self.tag_corpus(documents)
        
        # 2. Compute POS distribution
        pos_stats = self.compute_pos_distribution(tagged_corpus)
        
        # 3. Extract patterns
        patterns = self.extract_pos_patterns(tagged_corpus)
        
        # 4. Compare categories
        category_comparison = self.compare_categories_pos(documents, category_names, target_labels)
        
        # Prepare results
        results = {
            'tagged_corpus_sample': tagged_corpus[:10],  # Store sample
            'pos_distribution': pos_stats['pos_distribution'],
            'top_words_by_pos': pos_stats['top_words_by_pos'],
            'extracted_patterns': patterns,
            'category_comparison': category_comparison,
            'total_tagged_documents': len(tagged_corpus),
            'pos_tagger_used': 'spaCy' if self.use_spacy else 'NLTK'
        }
        
        # Log results
        self.logger.info(f"\n✓ POS analysis completed:")
        self.logger.info(f"  Tagged documents: {len(tagged_corpus)}")
        self.logger.info(f"  Total POS tags: {sum(pos_stats['pos_distribution'].values()):,}")
        self.logger.info(f"  POS tagger used: {results['pos_tagger_used']}")
        
        # Show top POS tags
        sorted_pos = sorted(pos_stats['pos_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info("\nTop 10 POS tags:")
        for pos, count in sorted_pos:
            self.logger.info(f"  {pos:10s}: {count:8,d}")
        
        return results
    
    def visualize(self, results: Dict, output_dir: str):
        """Generate visualizations for POS analysis"""
        self.logger.info("\nGenerating POS visualizations...")
        
        vis_dir = f"{output_dir}/visualizations/pos"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. POS Distribution Bar Chart
        pos_dist = results['pos_distribution']
        sorted_pos = sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:15]
        pos_tags, pos_counts = zip(*sorted_pos)
        
        plt.figure(figsize=(14, 7))
        colors = plt.cm.Set3(np.linspace(0, 1, len(pos_tags)))
        bars = plt.bar(pos_tags, pos_counts, color=colors)
        plt.title('POS Tag Distribution (Top 15)', fontsize=16, fontweight='bold')
        plt.xlabel('POS Tag', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, pos_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/pos_distribution.png', dpi=300)
        plt.close()
        
        # 2. Top Words by POS
        top_words = results['top_words_by_pos']
        if top_words:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            for idx, (pos, words_counts) in enumerate(top_words.items()):
                if idx >= 3:
                    break
                
                words, counts = zip(*words_counts[:10])
                
                ax = axes[idx]
                colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
                bars = ax.barh(words, counts, color=colors)
                ax.set_title(f'Top 10 {pos}s', fontsize=14, fontweight='bold')
                ax.set_xlabel('Frequency', fontsize=11)
                ax.invert_yaxis()  # Highest at top
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                           f' {count:,}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{vis_dir}/top_words_by_pos.png', dpi=300)
            plt.close()
        
        # 3. POS Patterns
        patterns = results['extracted_patterns']
        if patterns:
            fig, axes = plt.subplots(1, len(patterns), figsize=(5*len(patterns), 6))
            if len(patterns) == 1:
                axes = [axes]
            
            for idx, (pattern, items) in enumerate(patterns.items()):
                if idx >= len(axes):
                    break
                
                if items:
                    words, counts = zip(*items[:10])
                    ax = axes[idx]
                    colors = plt.cm.Paired(np.linspace(0, 1, len(words)))
                    bars = ax.barh(words, counts, color=colors)
                    ax.set_title(f'Top {pattern} Patterns', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Frequency', fontsize=10)
                    ax.invert_yaxis()
                    
                    for bar, count in zip(bars, counts):
                        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                               f' {count}', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{vis_dir}/pos_patterns.png', dpi=300)
            plt.close()
        
        # 4. Category Comparison (if available)
        if results['category_comparison']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            categories = list(results['category_comparison'].keys())
            
            # Plot 1: POS distribution comparison
            for idx, category in enumerate(categories[:2]):
                pos_dist = results['category_comparison'][category]['pos_distribution']
                top_pos = sorted(pos_dist.items(), key=lambda x: x[1], reverse=True)[:8]
                tags, counts = zip(*top_pos)
                
                ax = axes[idx]
                ax.bar(tags, counts, alpha=0.7)
                ax.set_title(f'POS Distribution - {category}', fontsize=12, fontweight='bold')
                ax.set_xlabel('POS Tag', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Top nouns comparison
            for idx, category in enumerate(categories[:2]):
                top_words = results['category_comparison'][category]['top_words_by_pos']
                if 'NOUN' in top_words:
                    nouns, counts = zip(*top_words['NOUN'][:8])
                    
                    ax = axes[idx + 2]
                    colors = plt.cm.spring(np.linspace(0, 1, len(nouns)))
                    ax.barh(nouns, counts, color=colors)
                    ax.set_title(f'Top Nouns - {category}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Frequency', fontsize=10)
                    ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f'{vis_dir}/category_comparison.png', dpi=300)
            plt.close()
        
        self.logger.info(f"✓ POS visualizations saved to {vis_dir}")