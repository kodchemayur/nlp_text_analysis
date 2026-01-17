"""
Tests for POS analysis module
"""
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pos_analysis import POSAnalyzer

class TestPOSAnalyzer(unittest.TestCase):
    def setUp(self):
        config = {
            'patterns_to_extract': ['ADJ+NOUN', 'NOUN+VERB'],
            'top_n_words': 5,
            'compare_categories': []
        }
        self.pos_analyzer = POSAnalyzer(config)
    
    def test_tag_document(self):
        text = "The quick brown fox jumps over the lazy dog."
        pos_tags = self.pos_analyzer.tag_document(text)
        
        self.assertIsInstance(pos_tags, list)
        self.assertGreater(len(pos_tags), 0)
        
        # Check format
        for word, pos in pos_tags:
            self.assertIsInstance(word, str)
            self.assertIsInstance(pos, str)
    
    def test_compute_pos_distribution(self):
        tagged_corpus = [
            [("The", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN")],
            [("jumps", "VERB"), ("over", "ADP"), ("the", "DET"), ("lazy", "ADJ"), ("dog", "NOUN")]
        ]
        
        distribution = self.pos_analyzer.compute_pos_distribution(tagged_corpus)
        
        self.assertIn('pos_distribution', distribution)
        self.assertIn('top_words_by_pos', distribution)
        
        pos_dist = distribution['pos_distribution']
        self.assertGreater(len(pos_dist), 0)
        
        # Check counts
        total_tags = sum(pos_dist.values())
        self.assertGreater(total_tags, 0)
        
        # Check top words
        top_words = distribution['top_words_by_pos']
        self.assertIsInstance(top_words, dict)
    
    def test_extract_patterns(self):
        tagged_corpus = [
            [("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN"), ("jumps", "VERB")],
            [("lazy", "ADJ"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")]
        ]
        
        patterns = self.pos_analyzer.extract_pos_patterns(tagged_corpus)
        
        self.assertIn('ADJ+NOUN', patterns)
        
        # Should find patterns
        adj_noun_patterns = patterns.get('ADJ+NOUN', [])
        self.assertGreater(len(adj_noun_patterns), 0)
    
    def test_analyze_method(self):
        # Create sample documents
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "A red apple fell from the green tree."
        ]
        categories = ["test", "test"]
        targets = [0, 0]
        
        results = self.pos_analyzer.analyze(documents, categories, targets)
        
        # Check results structure
        self.assertIn('pos_distribution', results)
        self.assertIn('top_words_by_pos', results)
        self.assertIn('extracted_patterns', results)
        self.assertIn('category_comparison', results)
        
        # Check data types
        self.assertIsInstance(results['pos_distribution'], dict)
        self.assertIsInstance(results['extracted_patterns'], dict)
        
        # Should have tagged some documents
        self.assertGreater(results['total_tagged_documents'], 0)

if __name__ == '__main__':
    unittest.main()