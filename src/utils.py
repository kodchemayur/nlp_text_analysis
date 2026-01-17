"""
Utility functions for NLP Pipeline
"""
import os
import pickle
import yaml
import logging
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def setup_logging(log_file="nlp_pipeline.log"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return {}

def save_results(data: Any, filename: str, output_dir="results"):
    """Save results to pickle file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logging.info(f"✓ Results saved to {filepath}")
    return filepath

def load_results(filename: str, output_dir="results") -> Any:
    """Load results from pickle file"""
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def create_output_dir(base_dir="output"):
    """Create output directory with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    return output_dir

def plot_bar(data: Dict, title: str, xlabel: str, ylabel: str, 
             filename: str, rotation=45, figsize=(12, 6)):
    """Create bar plot"""
    plt.figure(figsize=figsize)
    
    # Get top items
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)[:20]
    keys = [str(k) for k, _ in items]
    values = [v for _, v in items]
    
    bars = plt.bar(keys, values)
    plt.xticks(rotation=rotation, ha='right')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def print_top_items(items: List, title: str, top_n: int = 10):
    """Print top N items with formatting"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    for i, (item, count) in enumerate(items[:top_n], 1):
        if isinstance(item, tuple):
            item_str = ' '.join(item)
        else:
            item_str = str(item)
        print(f"{i:2d}. {item_str:40s} : {count:>8,d}")