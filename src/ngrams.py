"""
N-GRAM LANGUAGE MODELING MODULE
Part 2 of Assignment: Unigrams, Bigrams, Trigrams, Smoothing, Sentence Probability
"""
import numpy as np
from collections import Counter, defaultdict
from nltk import ngrams as nltk_ngrams
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math
import os

class NGramsAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.n_values = config.get('n_values', [1, 2, 3])
        self.smoothing = config.get('smoothing', 'laplace')
        self.top_k = config.get('top_k', 10)
        self.min_freq = config.get('min_freq', 5)
        self.logger = logging.getLogger(__name__)
        
        # Store models
        self.ngram_counts = {}
        self.vocabulary = set()
        self.total_tokens = 0
    
    def generate_ngrams(self, tokens_list: List[List[str]]) -> Dict[int, List]:
        """Generate n-grams for different n values"""
        all_ngrams = {}
        
        # Flatten tokens for statistics
        all_tokens = [token for tokens in tokens_list for token in tokens]
        self.total_tokens = len(all_tokens)
        self.vocabulary = set(all_tokens)
        
        self.logger.info(f"Generating n-grams from {self.total_tokens:,} tokens...")
        
        for n in self.n_values:
            self.logger.info(f"  Generating {n}-grams...")
            ngram_list = []
            
            for tokens in tokens_list:
                if len(tokens) >= n:
                    doc_ngrams = list(nltk_ngrams(tokens, n))
                    ngram_list.extend(doc_ngrams)
            
            all_ngrams[n] = ngram_list
            
            # Count frequencies
            self.ngram_counts[n] = Counter(ngram_list)
            
            # Log stats
            total_ngrams = len(ngram_list)
            unique_ngrams = len(self.ngram_counts[n])
            self.logger.info(f"    Total {n}-grams: {total_ngrams:,}")
            self.logger.info(f"    Unique {n}-grams: {unique_ngrams:,}")
        
        return all_ngrams
    
    def compute_frequency_distributions(self) -> Dict[int, List[Tuple]]:
        """Compute and sort n-gram frequencies"""
        freq_distributions = {}
        
        for n, counts in self.ngram_counts.items():
            sorted_ngrams = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            freq_distributions[n] = sorted_ngrams
            
            # Log top 5
            top_5 = sorted_ngrams[:5]
            self.logger.info(f"  Top 5 {n}-grams:")
            for ngram, freq in top_5:
                ngram_str = ' '.join(ngram) if isinstance(ngram, tuple) else str(ngram)
                self.logger.info(f"    '{ngram_str}': {freq:,}")
        
        return freq_distributions
    
    def apply_smoothing(self, ngram: Tuple, ngram_count: int, 
                       total_ngrams: int, vocab_size: int) -> float:
        """
        Implement one smoothing technique as required
        Options: 'laplace' (add-one), 'add_k', or 'none'
        """
        if self.smoothing == 'laplace':
            # Laplace (Add-one) smoothing
            return (ngram_count + 1) / (total_ngrams + vocab_size)
        
        elif self.smoothing == 'add_k':
            # Add-k smoothing (k=0.5)
            k = 0.5
            return (ngram_count + k) / (total_ngrams + k * vocab_size)
        
        else:
            # No smoothing (for comparison)
            return ngram_count / total_ngrams if total_ngrams > 0 else 0
    
    def calculate_sentence_probability(self, sentence_tokens: List[str]) -> float:
        """
        Calculate probability of a sentence using bigrams
        As required in assignment
        """
        if len(sentence_tokens) < 2:
            return 0.0
        
        log_prob = 0.0
        vocab_size = len(self.vocabulary)
        bigram_counts = self.ngram_counts[2]
        total_bigrams = sum(bigram_counts.values())
        
        # Add start and end tokens
        sentence_tokens = ['<s>'] + sentence_tokens + ['</s>']
        
        for i in range(len(sentence_tokens) - 1):
            bigram = (sentence_tokens[i], sentence_tokens[i + 1])
            bigram_count = bigram_counts.get(bigram, 0)
            
            # Apply smoothing
            prob = self.apply_smoothing(
                bigram, bigram_count, total_bigrams, vocab_size
            )
            
            # Use log probability to avoid underflow
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # Small epsilon for zero probability
        
        return math.exp(log_prob) if log_prob != 0 else 0
    
    def calculate_pmi(self, bigram_counts: Dict, unigram_counts: Dict) -> Dict[Tuple, float]:
        """Calculate Pointwise Mutual Information for bigrams"""
        pmi_scores = {}
        total_bigrams = sum(bigram_counts.values())
        
        for bigram, bigram_count in bigram_counts.items():
            if bigram_count < self.min_freq:
                continue
            
            word1, word2 = bigram
            word1_count = unigram_counts.get((word1,), 0)
            word2_count = unigram_counts.get((word2,), 0)
            
            if word1_count == 0 or word2_count == 0:
                continue
            
            # Calculate PMI: log(P(x,y) / (P(x)*P(y)))
            p_bigram = bigram_count / total_bigrams
            p_word1 = word1_count / self.total_tokens
            p_word2 = word2_count / self.total_tokens
            
            pmi = math.log(p_bigram / (p_word1 * p_word2))
            pmi_scores[bigram] = pmi
        
        return pmi_scores
    
    def get_top_informative_bigrams(self) -> List[Tuple]:
        """
        Identify the top 10 most informative bigrams using PMI
        As required in assignment
        """
        bigram_counts = self.ngram_counts[2]
        unigram_counts = self.ngram_counts[1]
        
        pmi_scores = self.calculate_pmi(bigram_counts, unigram_counts)
        
        # Sort by PMI score (descending)
        sorted_bigrams = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_k = min(self.top_k, len(sorted_bigrams))
        
        self.logger.info(f"Top {top_k} most informative bigrams (PMI):")
        for i, (bigram, pmi) in enumerate(sorted_bigrams[:top_k], 1):
            bigram_str = ' '.join(bigram)
            self.logger.info(f"  {i:2d}. {bigram_str:30s}: PMI = {pmi:.3f}")
        
        return sorted_bigrams[:self.top_k]
    
    def analyze(self, tokens_list: List[List[str]]) -> Dict:
        """
        Complete n-gram analysis pipeline
        Returns all results as dictionary
        """
        self.logger.info("\nStarting N-gram analysis...")
        
        # 1. Generate n-grams
        all_ngrams = self.generate_ngrams(tokens_list)
        
        # 2. Compute frequency distributions
        freq_distributions = self.compute_frequency_distributions()
        
        # 3. Get informative bigrams
        informative_bigrams = self.get_top_informative_bigrams()
        
        # 4. Calculate sentence probabilities for example sentences
        example_sentences = [
            ["computer", "graphics", "image", "processing"],
            ["space", "shuttle", "mission", "nasa"],
            ["baseball", "game", "team", "players"],
            ["medical", "research", "patient", "treatment"],
            ["government", "political", "policy", "decision"]
        ]
        
        sentence_probs = {}
        for sentence in example_sentences:
            prob = self.calculate_sentence_probability(sentence)
            sentence_probs[' '.join(sentence)] = prob
        
        self.logger.info("\nSentence probabilities (using bigrams with smoothing):")
        for sentence, prob in sentence_probs.items():
            self.logger.info(f"  '{sentence}': {prob:.2e}")
        
        # Prepare comprehensive results
        results = {
            'ngram_counts': self.ngram_counts,
            'freq_distributions': freq_distributions,
            'informative_bigrams': informative_bigrams,
            'sentence_probabilities': sentence_probs,
            'vocabulary_size': len(self.vocabulary),
            'total_tokens': self.total_tokens,
            'smoothing_technique': self.smoothing
        }
        
        self.logger.info(f"\n✓ N-gram analysis completed:")
        self.logger.info(f"  Vocabulary size: {len(self.vocabulary):,}")
        self.logger.info(f"  Total tokens: {self.total_tokens:,}")
        self.logger.info(f"  Smoothing technique: {self.smoothing}")
        
        return results
    
    def visualize(self, results: Dict, output_dir: str):
        """Generate visualizations for n-gram analysis"""
        self.logger.info("\nGenerating n-gram visualizations...")
        
        vis_dir = f"{output_dir}/visualizations/ngrams"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Plot top unigrams
        top_unigrams = results['freq_distributions'][1][:20]
        unigrams = [str(u[0]) for u, _ in top_unigrams]
        counts = [c for _, c in top_unigrams]
        
        plt.figure(figsize=(14, 7))
        bars = plt.bar(unigrams, counts, color='skyblue', alpha=0.8)
        plt.title('Top 20 Unigrams in 20 Newsgroups', fontsize=16, fontweight='bold')
        plt.xlabel('Unigram', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/top_unigrams.png', dpi=300)
        plt.close()
        
        # 2. Plot top bigrams
        top_bigrams = results['freq_distributions'][2][:20]
        bigram_strs = [' '.join(b) for b, _ in top_bigrams]
        bigram_counts = [c for _, c in top_bigrams]
        
        plt.figure(figsize=(14, 7))
        bars = plt.bar(bigram_strs, bigram_counts, color='lightcoral', alpha=0.8)
        plt.title('Top 20 Bigrams in 20 Newsgroups', fontsize=16, fontweight='bold')
        plt.xlabel('Bigram', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        for bar, count in zip(bars, bigram_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/top_bigrams.png', dpi=300)
        plt.close()
        
        # 3. Plot informative bigrams (PMI)
        informative = results['informative_bigrams'][:15]
        if informative:
            bigram_strs = [' '.join(b) for b, _ in informative]
            pmi_scores = [s for _, s in informative]
            
            plt.figure(figsize=(14, 7))
            colors = plt.cm.viridis(np.linspace(0, 1, len(bigram_strs)))
            bars = plt.bar(bigram_strs, pmi_scores, color=colors, alpha=0.8)
            plt.title('Top 15 Most Informative Bigrams (PMI)', fontsize=16, fontweight='bold')
            plt.xlabel('Bigram', fontsize=12)
            plt.ylabel('PMI Score', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            for bar, score in zip(bars, pmi_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{score:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{vis_dir}/informative_bigrams.png', dpi=300)
            plt.close()
        
        # 4. N-gram comparison
        ngram_types = ['Unigrams', 'Bigrams', 'Trigrams']
        ngram_counts = [
            len(results['freq_distributions'][1]),
            len(results['freq_distributions'][2]),
            len(results['freq_distributions'][3])
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ngram_types, ngram_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Number of Unique N-grams', fontsize=14, fontweight='bold')
        plt.xlabel('N-gram Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        for bar, count in zip(bars, ngram_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/ngram_counts.png', dpi=300)
        plt.close()
        
        self.logger.info(f"✓ N-gram visualizations saved to {vis_dir}")