"""
Tests for preprocessing module
"""
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import TextPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        config = {
            'use_spacy': False,
            'lemmatize': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'min_token_length': 2
        }
        self.preprocessor = TextPreprocessor(config)
    
    def test_clean_text(self):
        text = "Hello World! 123 Test."
        cleaned = self.preprocessor.clean_text(text)
        self.assertEqual(cleaned, "hello world test")
    
    def test_clean_text_url(self):
        text = "Visit http://example.com for info"
        cleaned = self.preprocessor.clean_text(text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("example.com", cleaned)
    
    def test_tokenize(self):
        text = "hello world test"
        tokens = self.preprocessor.tokenize(text)
        self.assertEqual(tokens, ["hello", "world", "test"])
    
    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "test", "document"]
        filtered = self.preprocessor.remove_stopwords(tokens)
        self.assertNotIn("this", filtered)
        self.assertNotIn("is", filtered)
        self.assertNotIn("a", filtered)
        self.assertIn("test", filtered)
        self.assertIn("document", filtered)
    
    def test_lemmatize_tokens(self):
        tokens = ["running", "cats", "better"]
        lemmatized = self.preprocessor.lemmatize_tokens(tokens)
        self.assertIn("run", lemmatized)  # running -> run
        self.assertIn("cat", lemmatized)  # cats -> cat
        self.assertIn("good", lemmatized)  # better -> good
    
    def test_full_preprocess(self):
        text = "The quick brown foxes are jumping over 2 lazy dogs!"
        cleaned, tokens = self.preprocessor.preprocess_text(text)
        
        self.assertTrue(isinstance(cleaned, str))
        self.assertTrue(isinstance(tokens, list))
        self.assertTrue(len(tokens) > 0)
        
        # Check no numbers
        self.assertNotIn("2", cleaned)
        
        # Check all tokens meet minimum length
        self.assertTrue(all(len(token) >= 2 for token in tokens))
        
        # Check stopwords removed
        self.assertNotIn("the", tokens)
        self.assertNotIn("are", tokens)
        self.assertNotIn("over", tokens)
    
    def test_empty_text(self):
        text = ""
        cleaned, tokens = self.preprocessor.preprocess_text(text)
        self.assertEqual(cleaned, "")
        self.assertEqual(tokens, [])
    
    def test_none_text(self):
        text = None
        cleaned, tokens = self.preprocessor.preprocess_text(text)
        self.assertEqual(cleaned, "")
        self.assertEqual(tokens, [])

if __name__ == '__main__':
    unittest.main()