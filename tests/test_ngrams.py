"""
Tests for n-grams module
"""
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ngrams import NGramsAnalyzer

class TestNGrams(unittest.TestCase):
    def setUp(self):
        config = {
            'n_values': [1, 2, 3],
            'smoothing': 'laplace',
            'top_k': 10,
            'min_freq': 2
        }
        self.ngram_analyzer = NGramsAnalyzer(config)
        
        # Sample tokens
        self.sample_tokens = [
            ["the", "quick", "brown", "fox"],
            ["jumps", "over", "the", "lazy", "dog"],
            ["the", "quick", "brown", "fox", "jumps", "again"]
        ]
    
    def test_generate_ngrams(self):
        ngrams = self.ngram_analyzer.generate_ngrams(self.sample_tokens)
        
        self.assertIn(1, ngrams)
        self.assertIn(2, ngrams)
        self.assertIn(3, ngrams)
        
        # Check counts
        self.assertGreater(len(ngrams[1]), 0)
        self.assertGreater(len(ngrams[2]), 0)
        self.assertGreater(len(ngrams[3]), 0)
    
    def test_compute_frequency_distributions(self):
        self.ngram_analyzer.generate_ngrams(self.sample_tokens)
        freq_dists = self.ngram_analyzer.compute_frequency_distributions()
        
        self.assertIn(1, freq_dists)
        self.assertIn(2, freq_dists)
        self.assertIn(3, freq_dists)
        
        # Check sorting (should be descending)
        for n, dist in freq_dists.items():
            if len(dist) > 1:
                counts = [count for _, count in dist]
                self.assertTrue(all(counts[i] >= counts[i+1] for i in range(len(counts)-1)))
    
    def test_smoothing(self):
        vocab_size = 10
        total_ngrams = 100
        
        # Test Laplace smoothing
        prob = self.ngram_analyzer.apply_smoothing(
            ("test", "ngram"), 5, total_ngrams, vocab_size
        )
        
        self.assertGreater(prob, 0)
        self.assertLess(prob, 1)
        
        # Test zero count
        prob_zero = self.ngram_analyzer.apply_smoothing(
            ("unknown", "ngram"), 0, total_ngrams, vocab_size
        )
        self.assertGreater(prob_zero, 0)  # Should be positive with smoothing
    
    def test_sentence_probability(self):
        # First generate n-grams
        self.ngram_analyzer.generate_ngrams(self.sample_tokens)
        
        # Test sentence
        sentence = ["the", "quick", "brown", "fox"]
        prob = self.ngram_analyzer.calculate_sentence_probability(sentence)
        
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 1)
    
    def test_pmi_calculation(self):
        # Create sample counts
        unigram_counts = {("the",): 10, ("quick",): 5, ("brown",): 3, ("fox",): 4}
        bigram_counts = {("the", "quick"): 5, ("quick", "brown"): 3, ("brown", "fox"): 2}
        
        self.ngram_analyzer.total_tokens = 100
        
        pmi_scores = self.ngram_analyzer.calculate_pmi(bigram_counts, unigram_counts)
        
        # Should have PMI scores for bigrams
        self.assertGreater(len(pmi_scores), 0)
        
        for bigram, pmi in pmi_scores.items():
            self.assertIsInstance(pmi, float)
    
    def test_get_top_informative_bigrams(self):
        # First generate n-grams
        self.ngram_analyzer.generate_ngrams(self.sample_tokens)
        
        informative = self.ngram_analyzer.get_top_informative_bigrams()
        
        # Should return list of tuples
        self.assertIsInstance(informative, list)
        
        if informative:
            for bigram, pmi in informative:
                self.assertIsInstance(bigram, tuple)
                self.assertIsInstance(pmi, float)

if __name__ == '__main__':
    unittest.main()