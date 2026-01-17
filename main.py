#!/usr/bin/env python3
"""
MAIN ENTRY POINT - NLP Pipeline Assignment
Follows assignment requirements exactly
"""
import argparse
import yaml
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import TextPreprocessor
from ngrams import NGramsAnalyzer
from pos_analysis import POSAnalyzer
from topic_modeling import TopicModeler

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("nlp_pipeline.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return {}

def save_results(data, filename, output_dir="results"):
    """Save results to pickle file"""
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"Results saved to {filepath}")
    return filepath

def load_results(filename, output_dir="results"):
    """Load results from pickle file"""
    import pickle
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def create_output_dir():
    """Create output directory"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    return output_dir

def main():
    # Setup
    parser = argparse.ArgumentParser(description="NLP Pipeline Assignment - 20 Newsgroups Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--run-all", action="store_true", help="Run all pipeline steps")
    parser.add_argument("--step", type=str, 
                       choices=["preprocess", "ngram", "pos", "topic", "all"], 
                       help="Run specific step")
    parser.add_argument("--output", type=str, help="Output directory name")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("NLP Pipeline Assignment - 20 Newsgroups Dataset")
    logger.info("Following assignment requirements exactly")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = args.output if args.output else create_output_dir()
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize components
    preprocessor = TextPreprocessor(config.get('preprocessing', {}))
    
    # Load 20 Newsgroups dataset
    logger.info("\n" + "="*70)
    logger.info("LOADING 20 NEWSCROUPS DATASET (FULL DATASET)")
    logger.info("="*70)
    
    try:
        # Load full dataset as per assignment requirements
        documents, category_names, target_labels = preprocessor.load_20newsgroups_full()
        logger.info(f"✓ Loaded {len(documents)} documents (FULL DATASET)")
        logger.info(f"✓ Categories: {category_names}")
        
        # Save dataset info
        dataset_info = {
            'num_documents': len(documents),
            'categories': category_names,
            'category_distribution': dict(zip(*np.unique(target_labels, return_counts=True)))
        }
        save_results(dataset_info, 'dataset_info.pkl', f"{output_dir}/results")
        
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        return
    
    # Run all steps if --run-all or --step all
    run_all_steps = args.run_all or args.step == "all"
    
    # PART 1: Text Preprocessing
    if run_all_steps or args.step == "preprocess" or args.step is None:
        logger.info("\n" + "="*70)
        logger.info("PART 1: TEXT PREPROCESSING")
        logger.info("="*70)
        logger.info("Tasks: Lowercasing, Tokenization, Stopword removal, Lemmatization")
        
        cleaned_docs, tokens_list = preprocessor.preprocess_corpus(documents)
        
        # Save cleaned corpus
        save_results({
            'cleaned_documents': cleaned_docs,
            'tokens_list': tokens_list,
            'category_names': category_names,
            'target_labels': target_labels
        }, 'cleaned_corpus.pkl', f"{output_dir}/results")
        
        logger.info(f"✓ Preprocessing completed. Saved {len(cleaned_docs)} documents")
    
    # Load cleaned data for subsequent steps
    try:
        data = load_results('cleaned_corpus.pkl', f"{output_dir}/results")
        cleaned_docs = data['cleaned_documents']
        tokens_list = data['tokens_list']
        category_names = data['category_names']
        target_labels = data['target_labels']
    except:
        logger.warning("Could not load cleaned corpus, preprocessing might have failed")
        return
    
    # PART 2: N-gram Language Modeling
    if run_all_steps or args.step == "ngram":
        logger.info("\n" + "="*70)
        logger.info("PART 2: N-GRAM LANGUAGE MODELING")
        logger.info("="*70)
        logger.info("Tasks: Generate unigrams/bigrams/trigrams, frequency distributions")
        logger.info("       Implement smoothing, calculate sentence probability")
        logger.info("       Identify top 10 most informative bigrams")
        
        ngram_analyzer = NGramsAnalyzer(config.get('ngrams', {}))
        ngram_results = ngram_analyzer.analyze(tokens_list)
        
        # Save results
        save_results(ngram_results, 'ngram_results.pkl', f"{output_dir}/results")
        
        # Generate visualizations
        ngram_analyzer.visualize(ngram_results, output_dir)
        
        logger.info("✓ N-gram analysis completed")
        logger.info(f"  Vocabulary size: {ngram_results['vocabulary_size']:,}")
        logger.info(f"  Top informative bigrams saved")
    
    # PART 3: POS Tagging
    if run_all_steps or args.step == "pos":
        logger.info("\n" + "="*70)
        logger.info("PART 3: POS TAGGING")
        logger.info("="*70)
        logger.info("Tasks: Apply POS tagging, compute POS distribution")
        logger.info("       Top 10 nouns/verbs/adjectives, extract POS patterns")
        logger.info("       Compare POS distributions across categories")
        
        pos_analyzer = POSAnalyzer(config.get('pos', {}))
        pos_results = pos_analyzer.analyze(cleaned_docs, category_names, target_labels)
        
        # Save results
        save_results(pos_results, 'pos_results.pkl', f"{output_dir}/results")
        
        # Generate visualizations
        pos_analyzer.visualize(pos_results, output_dir)
        
        logger.info("✓ POS analysis completed")
    
    # PART 4: Topic Modeling
    if run_all_steps or args.step == "topic":
        logger.info("\n" + "="*70)
        logger.info("PART 4: TOPIC MODELING")
        logger.info("="*70)
        logger.info("Tasks: Convert to document-term matrix, apply LDA")
        logger.info("       Experiment with 3 values of K, list top 10 words per topic")
        logger.info("       Assign dominant topic, visualize topics")
        if config.get('topic', {}).get('compare_lda_nmf', False):
            logger.info("       BONUS: Compare LDA with NMF")
        
        topic_modeler = TopicModeler(config.get('topic', {}))
        topic_results = topic_modeler.analyze(cleaned_docs, category_names)
        
        # Save results
        save_results(topic_results, 'topic_results.pkl', f"{output_dir}/results")
        
        # Generate visualizations
        topic_modeler.visualize(topic_results, output_dir)
        
        logger.info("✓ Topic modeling completed")
        logger.info(f"  Best K: {topic_results['best_k']}")
    
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"All results saved in: {output_dir}")
    logger.info(f"Check {output_dir}/visualizations/ for plots")
    logger.info(f"Check {output_dir}/results/ for saved data")
    logger.info("="*70)

if __name__ == "__main__":
    import numpy as np
    main()