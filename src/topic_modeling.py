"""
TOPIC MODELING MODULE
Part 4 of Assignment: LDA, experiment with K values, visualize topics
Bonus: Compare LDA with NMF
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import GridSearchCV
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import logging
from typing import List, Dict, Tuple, Any
import pickle
import os

class TopicModeler:
    def __init__(self, config: Dict):
        self.config = config
        self.n_topics_range = config.get('n_topics_range', [5, 8, 10])
        self.max_features = config.get('max_features', 1000)
        self.min_df = config.get('min_df', 2)
        self.max_df = config.get('max_df', 0.95)
        self.should_compare_models = config.get('compare_lda_nmf', False)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.vectorizer = None
        self.dtm = None
        self.documents = None  # Store documents for NMF
        self.lda_model = None
        self.nmf_model = None
        self.best_k = None
    
    def create_document_term_matrix(self, documents: List[str]):
        """Create document-term matrix"""
        self.logger.info("Creating document-term matrix...")
        
        # Store documents for NMF training
        self.documents = documents
        
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True
        )
        
        self.dtm = self.vectorizer.fit_transform(documents)
        
        self.logger.info(f"✓ DTM created: {self.dtm.shape[0]} documents x {self.dtm.shape[1]} features")
        return self.dtm
    
    def train_lda(self, n_topics: int = 10, random_state: int = 42):
        """Train LDA model"""
        self.logger.info(f"Training LDA with {n_topics} topics...")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            learning_method='online',
            max_iter=20,
            learning_offset=50.0,
            verbose=0
        )
        
        self.lda_model.fit(self.dtm)
        
        self.logger.info(f"✓ LDA trained with {n_topics} topics")
        return self.lda_model
    
    def train_nmf(self, n_topics: int = 10, random_state: int = 42):
        """Train NMF model (for comparison) - FIXED VERSION"""
        self.logger.info(f"Training NMF with {n_topics} topics...")
        
        # Check if we have documents
        if self.documents is None:
            self.logger.error("No documents available for NMF training")
            return None
        
        # Use TF-IDF for NMF (usually works better)
        # Use different parameters than LDA to avoid pruning issues
        tfidf_vectorizer = TfidfVectorizer(
            max_features=min(800, len(self.documents)),  # Smaller vocabulary
            min_df=1,  # Lower min_df
            max_df=0.99,  # Higher max_df  
            stop_words='english',
            lowercase=True
        )
        
        try:
            # Transform ACTUAL DOCUMENTS to TF-IDF (not vocabulary words!)
            tfidf_dtm = tfidf_vectorizer.fit_transform(self.documents)
            
            if tfidf_dtm.shape[1] == 0:
                self.logger.warning("TF-IDF resulted in no features. Trying with even looser parameters...")
                # Try with even looser parameters
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=500,
                    min_df=1,
                    max_df=1.0,  # Allow all terms
                    stop_words='english',
                    lowercase=True
                )
                tfidf_dtm = tfidf_vectorizer.fit_transform(self.documents)
            
            if tfidf_dtm.shape[1] == 0:
                self.logger.error("Still no features after adjusting parameters. Skipping NMF.")
                return None
            
            self.logger.info(f"TF-IDF DTM shape: {tfidf_dtm.shape}")
            
            self.nmf_model = NMF(
                n_components=n_topics,
                random_state=random_state,
                max_iter=500,
                verbose=0,
                alpha=0.1,
                l1_ratio=0.5
            )
            
            self.nmf_model.fit(tfidf_dtm)
            
            # Store the TF-IDF vectorizer for getting feature names
            self.tfidf_vectorizer = tfidf_vectorizer
            
            self.logger.info(f"✓ NMF trained with {n_topics} topics")
            return self.nmf_model
        
        except Exception as e:
            self.logger.error(f"NMF training failed: {e}")
            self.logger.info("Skipping NMF comparison")
            return None
    
    def get_topic_words(self, model, n_words: int = 10, model_type: str = 'lda'):
        """Get top words for each topic"""
        if model_type == 'lda':
            feature_names = self.vectorizer.get_feature_names_out()
        else:  # NMF
            if hasattr(self, 'tfidf_vectorizer'):
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
            else:
                self.logger.warning("No TF-IDF vectorizer found for NMF")
                return []
        
        topics = []
        if model_type == 'lda':
            for topic_idx, topic in enumerate(model.components_):
                top_word_indices = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                topic_weight = topic.sum()
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic[top_word_indices].tolist()
                })
        else:  # NMF
            for topic_idx, topic in enumerate(model.components_):
                top_word_indices = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': topic[top_word_indices].tolist()
                })
        
        return topics
    
    def assign_topics_to_documents(self, documents: List[str]):
        """Assign dominant topic to each document"""
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        # Get document-topic distribution
        doc_topic_dist = self.lda_model.transform(self.dtm)
        
        # Assign dominant topic
        dominant_topics = doc_topic_dist.argmax(axis=1)
        
        # Get topic confidence
        topic_confidence = doc_topic_dist.max(axis=1)
        
        # Prepare results
        document_topics = []
        for i, (topic, confidence) in enumerate(zip(dominant_topics, topic_confidence)):
            document_topics.append({
                'document_id': i,
                'dominant_topic': int(topic),
                'confidence': float(confidence),
                'topic_distribution': doc_topic_dist[i].tolist()
            })
        
        # Create summary
        topic_counts = np.bincount(dominant_topics)
        topic_summary = []
        for topic_id in range(len(topic_counts)):
            topic_summary.append({
                'topic_id': topic_id,
                'document_count': int(topic_counts[topic_id]),
                'percentage': float(topic_counts[topic_id] / len(documents) * 100)
            })
        
        return document_topics, topic_summary
    
    def calculate_coherence(self, n_topics: int, model_type: str = 'lda'):
        """Calculate topic coherence score"""
        # Prepare texts for gensim
        texts = [doc.split() for doc in self.documents[:500]]  # Sample for speed
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        if model_type == 'lda':
            # Train gensim LDA model
            gensim_lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=gensim_lda,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
        else:
            coherence_score = 0
        
        return coherence_score
    
    def find_optimal_topics(self):
        """Find optimal number of topics using coherence scores"""
        self.logger.info("\nFinding optimal number of topics (experimenting with 3 K values)...")
        
        coherence_scores = {}
        
        for n_topics in self.n_topics_range:
            # Train LDA
            lda_model = self.train_lda(n_topics)
            
            # Calculate coherence
            coherence = self.calculate_coherence(n_topics, 'lda')
            coherence_scores[n_topics] = coherence
            
            self.logger.info(f"  K={n_topics:2d}: Coherence = {coherence:.4f}")
        
        # Find best K
        self.best_k = max(coherence_scores, key=coherence_scores.get)
        
        self.logger.info(f"\n✓ Optimal number of topics: K={self.best_k} (coherence: {coherence_scores[self.best_k]:.4f})")
        
        return coherence_scores
    
    def compare_lda_nmf(self, n_topics: int = 10):
        """Compare LDA and NMF models (BONUS task)"""
        self.logger.info("\nComparing LDA and NMF models (BONUS)...")
        
        # Train LDA
        lda_model = self.train_lda(n_topics)
        
        # Try to train NMF
        nmf_model = None
        nmf_topics = []
        
        try:
            nmf_model = self.train_nmf(n_topics)
            if nmf_model is not None:
                nmf_topics = self.get_topic_words(nmf_model, model_type='nmf')
        except Exception as e:
            self.logger.warning(f"NMF training failed: {e}")
            self.logger.info("Skipping NMF comparison")
        
        # Get LDA topic words
        lda_topics = self.get_topic_words(lda_model, model_type='lda')
        
        # Calculate metrics
        comparison = {
            'lda': {
                'topics': lda_topics,
                'perplexity': lda_model.perplexity(self.dtm) if hasattr(lda_model, 'perplexity') else None,
                'log_likelihood': lda_model.score(self.dtm) if hasattr(lda_model, 'score') else None
            },
            'nmf': {
                'topics': nmf_topics,
                'reconstruction_error': nmf_model.reconstruction_err_ if nmf_model and hasattr(nmf_model, 'reconstruction_err_') else None
            }
        }
        
        # Log comparison
        self.logger.info("\nLDA vs NMF Comparison:")
        self.logger.info(f"  LDA Perplexity: {comparison['lda']['perplexity']:.2f}" if comparison['lda']['perplexity'] else "  LDA Perplexity: N/A")
        
        if nmf_model:
            self.logger.info(f"  NMF Reconstruction Error: {comparison['nmf']['reconstruction_error']:.4f}" if comparison['nmf']['reconstruction_error'] else "  NMF Reconstruction Error: N/A")
            self.logger.info("  ✓ NMF comparison completed")
        else:
            self.logger.info("  NMF: Not available (training failed or skipped)")
        
        return comparison
    
    def visualize_topics_pyldavis(self, lda_model, output_dir: str):
        """Create interactive visualization using pyLDAvis"""
        try:
            self.logger.info("Creating pyLDAvis visualization...")
            
            # Prepare data for pyLDAvis
            texts = [doc.split() for doc in self.documents[:500]]  # Sample
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # Create gensim LDA model for visualization
            gensim_lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=lda_model.n_components,
                random_state=42,
                passes=10
            )
            
            # Prepare visualization
            vis_data = gensimvis.prepare(gensim_lda, corpus, dictionary)
            
            # Save to HTML
            vis_dir = f"{output_dir}/visualizations/topics"
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = f"{vis_dir}/lda_visualization.html"
            pyLDAvis.save_html(vis_data, vis_path)
            
            self.logger.info(f"✓ pyLDAvis visualization saved to {vis_path}")
            
            return vis_data
        except Exception as e:
            self.logger.warning(f"pyLDAvis visualization failed: {e}")
            return None
    
    def create_wordclouds(self, topics: List[Dict], model_type: str, output_dir: str):
        """Create word clouds for each topic"""
        self.logger.info(f"Creating word clouds for {model_type} topics...")
        
        vis_dir = f"{output_dir}/visualizations/topics/wordclouds"
        os.makedirs(vis_dir, exist_ok=True)
        
        for topic in topics:
            topic_id = topic['topic_id']
            words = topic['words']
            weights = topic.get('weights', [1] * len(words))
            
            # Create word frequencies dictionary
            word_freq = {word: weight for word, weight in zip(words, weights)}
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=50,
                contour_width=1,
                contour_color='steelblue'
            ).generate_from_frequencies(word_freq)
            
            # Plot
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Topic {topic_id} - {model_type.upper()}', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save
            filename = f'{vis_dir}/{model_type}_topic_{topic_id}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"✓ Word clouds saved for {len(topics)} topics")
    
    def analyze(self, documents: List[str], category_names: List[str] = None) -> Dict:
        """Complete topic modeling analysis pipeline"""
        self.logger.info("\nStarting Topic Modeling analysis...")
        
        # 1. Create DTM (also stores documents)
        self.create_document_term_matrix(documents)
        
        # 2. Find optimal number of topics (experiment with 3 values)
        coherence_scores = self.find_optimal_topics()
        
        # 3. Train final LDA model with best K
        final_lda = self.train_lda(self.best_k)
        
        # 4. Get topic words
        topic_words = self.get_topic_words(final_lda, n_words=15, model_type='lda')
        
        # 5. Assign topics to documents
        document_topics, topic_summary = self.assign_topics_to_documents(documents)
        
        # 6. Compare LDA vs NMF (BONUS) - FIXED
        model_comparison = {}
        if self.should_compare_models:
            model_comparison = self.compare_lda_nmf(self.best_k)
        
        # Prepare comprehensive results
        results = {
            'best_k': self.best_k,
            'coherence_scores': coherence_scores,
            'topic_words': topic_words,
            'document_topics': document_topics,
            'topic_summary': topic_summary,
            'model_comparison': model_comparison,
            'dtm_shape': self.dtm.shape,
            'vocabulary_size': len(self.vectorizer.get_feature_names_out()),
            'num_documents': len(documents)
        }
        
        # Log results
        self.logger.info(f"\n✓ Topic modeling completed:")
        self.logger.info(f"  Best K: {self.best_k}")
        self.logger.info(f"  Documents: {len(documents):,}")
        self.logger.info(f"  Vocabulary size: {results['vocabulary_size']:,}")
        
        # Show topic summary
        self.logger.info("\nTopic Distribution:")
        for topic_info in topic_summary:
            self.logger.info(f"  Topic {topic_info['topic_id']:2d}: {topic_info['document_count']:4d} docs ({topic_info['percentage']:.1f}%)")
        
        # Show top words for each topic
        self.logger.info("\nTop Words per Topic:")
        for topic in topic_words[:5]:  # Show first 5 topics
            self.logger.info(f"  Topic {topic['topic_id']:2d}: {', '.join(topic['words'][:5])}")
        
        return results
    
    def visualize(self, results: Dict, output_dir: str):
        """Generate all visualizations for topic modeling"""
        self.logger.info("\nGenerating topic modeling visualizations...")
        
        vis_dir = f"{output_dir}/visualizations/topics"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Coherence Scores Plot
        coherence_scores = results['coherence_scores']
        
        plt.figure(figsize=(10, 6))
        x = list(coherence_scores.keys())
        y = list(coherence_scores.values())
        
        plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')
        plt.scatter([results['best_k']], [coherence_scores[results['best_k']]], 
                   color='red', s=200, zorder=5, label=f'Best K={results["best_k"]}')
        
        plt.title('Topic Coherence Scores for Different K Values', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Topics (K)', fontsize=12)
        plt.ylabel('Coherence Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/coherence_scores.png', dpi=300)
        plt.close()
        
        # 2. Topic Word Importance
        topic_words = results['topic_words']
        n_topics = len(topic_words)
        
        # Create topic-word matrix for heatmap (only if reasonable size)
        if n_topics <= 15 and len(topic_words) > 0:
            all_words = set()
            for topic in topic_words:
                all_words.update(topic['words'][:8])  # Top 8 words per topic
            
            all_words = sorted(list(all_words))
            
            if len(all_words) <= 50:  # Only create heatmap if manageable size
                # Create importance matrix
                importance_matrix = np.zeros((len(all_words), n_topics))
                
                for topic in topic_words:
                    topic_id = topic['topic_id']
                    for word, weight in zip(topic['words'], topic.get('weights', [])):
                        if word in all_words:
                            word_idx = all_words.index(word)
                            importance_matrix[word_idx, topic_id] = weight
                
                # Plot heatmap
                plt.figure(figsize=(max(10, n_topics*1.5), max(8, len(all_words)*0.3)))
                sns.heatmap(importance_matrix, 
                           xticklabels=[f'Topic {i}' for i in range(n_topics)],
                           yticklabels=all_words,
                           cmap='YlOrRd',
                           linewidths=0.5)
                
                plt.title('Word Importance Across Topics', fontsize=14, fontweight='bold')
                plt.xlabel('Topics', fontsize=12)
                plt.ylabel('Words', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{vis_dir}/topic_word_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Document-Topic Distribution
        doc_topics = results['document_topics']
        dominant_topics = [doc['dominant_topic'] for doc in doc_topics]
        
        plt.figure(figsize=(10, 6))
        plt.hist(dominant_topics, bins=n_topics, alpha=0.7, edgecolor='black', color='lightgreen')
        plt.title('Distribution of Dominant Topics Across Documents', fontsize=14, fontweight='bold')
        plt.xlabel('Topic ID', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.xticks(range(n_topics))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/document_topic_distribution.png', dpi=300)
        plt.close()
        
        # 4. Create word clouds
        self.create_wordclouds(topic_words, 'lda', output_dir)
        
        # 5. Create pyLDAvis visualization
        if self.lda_model:
            self.visualize_topics_pyldavis(self.lda_model, output_dir)
        
        # 6. LDA vs NMF Comparison (if available)
        if 'model_comparison' in results and results['model_comparison'] and results['model_comparison']['nmf']['topics']:
            comparison = results['model_comparison']
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))
            
            # LDA Topics
            lda_words = [', '.join(topic['words'][:5]) for topic in comparison['lda']['topics'][:5]]
            axes[0].barh(range(len(lda_words)), [1] * len(lda_words))
            axes[0].set_yticks(range(len(lda_words)))
            axes[0].set_yticklabels([f'Topic {i}: {words}' for i, words in enumerate(lda_words)])
            axes[0].set_title('LDA - Top 5 Words per Topic', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Topics')
            axes[0].invert_yaxis()
            
            # NMF Topics
            nmf_words = [', '.join(topic['words'][:5]) for topic in comparison['nmf']['topics'][:5]]
            axes[1].barh(range(len(nmf_words)), [1] * len(nmf_words))
            axes[1].set_yticks(range(len(nmf_words)))
            axes[1].set_yticklabels([f'Topic {i}: {words}' for i, words in enumerate(nmf_words)])
            axes[1].set_title('NMF - Top 5 Words per Topic', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Topics')
            axes[1].invert_yaxis()
            
            plt.suptitle('LDA vs NMF Topic Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{vis_dir}/lda_vs_nmf_comparison.png', dpi=300)
            plt.close()
        
        # 7. Topic Proportions Pie Chart
        topic_summary = results['topic_summary']
        labels = [f'Topic {t["topic_id"]}' for t in topic_summary]
        sizes = [t['document_count'] for t in topic_summary]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab20c(np.linspace(0, 1, len(labels)))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Topic Distribution in Documents', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/topic_distribution_pie.png', dpi=300)
        plt.close()
        
        self.logger.info(f"✓ Topic modeling visualizations saved to {vis_dir}")