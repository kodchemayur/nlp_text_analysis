"""
Tests for topic modeling module
"""
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from topic_modeling import TopicModeler

class TestTopicModeling(unittest.TestCase):
    def setUp(self):
        config = {
            'n_topics_range': [3, 5],
            'max_features': 100,
            'min_df': 1,
            'max_df': 0.95,
            'compare_lda_nmf': False
        }
        self.topic_modeler = TopicModeler(config)
        
        # Sample documents
        self.sample_docs = [
            "computer science machine learning artificial intelligence",
            "biology chemistry physics scientific research",
            "football basketball sports games teams",
            "politics government election democracy",
            "art music painting creative expression"
        ]
    
    def test_create_document_term_matrix(self):
        dtm = self.topic_modeler.create_document_term_matrix(self.sample_docs)
        
        self.assertIsNotNone(dtm)
        self.assertEqual(dtm.shape[0], len(self.sample_docs))
        self.assertGreater(dtm.shape[1], 0)
        
        # Should have vectorizer
        self.assertIsNotNone(self.topic_modeler.vectorizer)
    
    def test_train_lda(self):
        # First create DTM
        self.topic_modeler.create_document_term_matrix(self.sample_docs)
        
        # Train LDA
        lda_model = self.topic_modeler.train_lda(n_topics=3)
        
        self.assertIsNotNone(lda_model)
        self.assertEqual(lda_model.n_components, 3)
        
        # Check components shape
        self.assertEqual(lda_model.components_.shape[0], 3)
        self.assertEqual(lda_model.components_.shape[1], self.topic_modeler.dtm.shape[1])
    
    def test_get_topic_words(self):
        # Setup
        self.topic_modeler.create_document_term_matrix(self.sample_docs)
        lda_model = self.topic_modeler.train_lda(n_topics=2)
        
        # Get topic words
        topics = self.topic_modeler.get_topic_words(lda_model, n_words=5, model_type='lda')
        
        self.assertIsInstance(topics, list)
        self.assertEqual(len(topics), 2)
        
        for topic in topics:
            self.assertIn('topic_id', topic)
            self.assertIn('words', topic)
            self.assertIn('weights', topic)
            
            self.assertEqual(len(topic['words']), 5)
            self.assertEqual(len(topic['weights']), 5)
            
            # Words should be strings
            for word in topic['words']:
                self.assertIsInstance(word, str)
    
    def test_assign_topics_to_documents(self):
        # Setup
        self.topic_modeler.create_document_term_matrix(self.sample_docs)
        self.topic_modeler.train_lda(n_topics=2)
        
        # Assign topics
        doc_topics, topic_summary = self.topic_modeler.assign_topics_to_documents(self.sample_docs)
        
        self.assertIsInstance(doc_topics, list)
        self.assertEqual(len(doc_topics), len(self.sample_docs))
        
        self.assertIsInstance(topic_summary, list)
        self.assertEqual(len(topic_summary), 2)
        
        for doc in doc_topics:
            self.assertIn('document_id', doc)
            self.assertIn('dominant_topic', doc)
            self.assertIn('confidence', doc)
            self.assertIn('topic_distribution', doc)
            
            self.assertGreaterEqual(doc['dominant_topic'], 0)
            self.assertLess(doc['dominant_topic'], 2)
            self.assertGreaterEqual(doc['confidence'], 0)
            self.assertLessEqual(doc['confidence'], 1)
            
            # Distribution should sum to ~1
            dist_sum = sum(doc['topic_distribution'])
            self.assertAlmostEqual(dist_sum, 1.0, places=2)
    
    def test_find_optimal_topics(self):
        # Setup DTM
        self.topic_modeler.create_document_term_matrix(self.sample_docs)
        
        # Find optimal topics
        coherence_scores = self.topic_modeler.find_optimal_topics()
        
        self.assertIsInstance(coherence_scores, dict)
        self.assertEqual(len(coherence_scores), len(self.topic_modeler.n_topics_range))
        
        # Should have best_k
        self.assertIsNotNone(self.topic_modeler.best_k)
        self.assertIn(self.topic_modeler.best_k, self.topic_modeler.n_topics_range)

if __name__ == '__main__':
    unittest.main()