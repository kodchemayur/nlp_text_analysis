# NLP Pipeline: Text Analysis of 20 Newsgroups Dataset

## 📚 Project Overview
This project implements a complete Natural Language Processing (NLP) pipeline to analyze the 20 Newsgroups dataset using n-gram models, POS tagging, and topic modeling.

**Objective:** Build an end-to-end NLP pipeline that preprocesses text, extracts linguistic patterns, and uncovers latent topics using various NLP techniques.

## 🏗️ Project Structure
nlp_assignment/
├── src/
│ ├── preprocessing.py # Text cleaning and normalization
│ ├── ngrams.py # N-gram language modeling
│ ├── pos_analysis.py # POS tagging and pattern extraction
│ └── topic_modeling.py # Topic modeling with LDA/NMF
├── tests/
│ ├── test_preprocessing.py
│ ├── test_ngrams.py
│ ├── test_pos.py
│ └── test_topic_modeling.py
├── main.py # Main pipeline runner
├── requirements.txt # Python dependencies
├── config.yaml # Configuration file
└── README.md # This file


## 🎯 Features
### Part 1: Text Preprocessing
- Lowercasing, tokenization, stopword removal
- Lemmatization/stemming
- Special character and number removal

### Part 2: N-gram Language Modeling
- Generate unigrams, bigrams, trigrams
- Compute frequency distributions
- Implement Laplace smoothing
- Calculate sentence probability
- Identify top 10 most informative bigrams (PMI)

### Part 3: POS Tagging
- Apply POS tagging using spaCy
- Compute POS tag distributions
- Extract syntactic patterns (Adj+Noun, Noun+Verb)
- Compare POS distributions across categories

### Part 4: Topic Modeling
- Convert corpus to document-term matrix
- Apply Latent Dirichlet Allocation (LDA)
- Experiment with 3 K values (5, 8, 10 topics)
- Select best model using coherence scores
- Visualize topics with word clouds and pyLDAvis

### Bonus Features
- Compare LDA with Non-negative Matrix Factorization (NMF)
- Evaluate topic coherence
- Generate comprehensive visualizations

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8+
- pip package manager

### 2. Clone Repository
bash
git clone [repository-url]
cd nlp_assignment

### 3. Install Dependencies
bash
pip install -r requirements.txt

### 4. Download NLTK Data

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

### 5. Download spaCy Model
bash
python -m spacy download en_core_web_sm

## 📊 Output & Results

The pipeline generates:

- Cleaned corpus (stored as pickle files)

- N-gram frequency distributions

- POS tag distributions and patterns

- Topic models with top words

Visualizations in output_YYYYMMDD_HHMMSS/:

- N-gram frequency plots

- POS distribution charts

- Topic word clouds

- pyLDAvis interactive visualization

- LDA vs NMF comparison

## 📁 Dataset

20 Newsgroups Dataset

20,000 newsgroup documents

Organized into 20 categories

Topics: computers, politics, sports, religion, etc.

Minimum 1,000 documents used as required
