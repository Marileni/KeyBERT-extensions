 
import json
import os
import unicodedata
import string

# Data splitting
from sklearn.model_selection import train_test_split

# Text & NLP
import spacy
import nltk
from nltk.corpus import stopwords

# Fuzzy matching
from thefuzz import fuzz

# Embeddings + semantic similarity
from sentence_transformers import SentenceTransformer, util
import torch


#############################
#  A) Global Setup/Models   #
#############################

# Download Greek stopwords if not already present
# (Safe to call multiple times; it will only download once)
nltk.download('stopwords')

# Load Greek stopwords
greek_stopwords = stopwords.words('greek')

# Load the Greek Spacy model
# Make sure you have installed it: "python -m spacy download el_core_news_sm"
nlp = spacy.load("el_core_news_sm")

# Sentence-BERT model for semantic similarity
base_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
sbert_model = SentenceTransformer(base_model_name)


##################################
#  B) Utility / Helper Functions #
##################################

def remove_greek_accents(text: str) -> str:
    """
    Removes Greek accents (tonos) from a string.
    """
    text_normalized = unicodedata.normalize('NFD', text)
    text_no_accents = ''.join(ch for ch in text_normalized if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', text_no_accents)


def normalize_greek(s: str) -> str:
    """
    Lowercases and removes Greek accents from a string.
    """
    s = s.lower()
    s_nfd = unicodedata.normalize('NFD', s)
    s_no_accents = ''.join(ch for ch in s_nfd if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', s_no_accents)


def preprocess_greek(text: str) -> list:
    """
    Lowercases, removes accents & punctuation, and filters out Greek stopwords.
    Returns a list of tokens.
    """
    text = text.lower()
    text = remove_greek_accents(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in greek_stopwords]
    return tokens


def generate_candidates(text: str, max_ngram: int = 4) -> list:
    """
    Generates candidate keywords by:
      1) Extracting noun chunks via SpaCy.
      2) Building n-grams up to 'max_ngram' from the preprocessed text.
    """
    doc = nlp(text)
    candidates = set()

    # 1) Collect noun chunks
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        chunk_text = remove_greek_accents(chunk_text)
        chunk_text = chunk_text.translate(str.maketrans("", "", string.punctuation))
        candidates.add(chunk_text.strip())

    # 2) Collect n-grams (up to max_ngram)
    tokens = preprocess_greek(text)
    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            candidates.add(ngram.strip())

    return list(candidates)


#######################################
#  C) Hybrid Matching for Labeling    #
#######################################

class HybridMatcher:
    """
    For each doc, we create a HybridMatcher on the gold keywords,
    then do exact/fuzzy checks and fall back to a semantic check if needed.
    """
    def __init__(self, gold_keywords, model, fuzzy_threshold=85, semantic_threshold=0.75):
        self.model = model
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.gold_data = []

        # Pre-encode all gold keywords for semantic checks
        for kw in gold_keywords:
            kw_norm = normalize_greek(kw.strip())
            kw_emb = model.encode(kw_norm, convert_to_tensor=True)
            self.gold_data.append((kw, kw_norm, kw_emb))

    def is_gold_candidate(self, cand: str) -> bool:
        """
        Check if candidate 'cand' matches any gold keyword
        via exact match, fuzzy match, or semantic match.
        """
        cand_norm = normalize_greek(cand.strip())
        cand_emb = None

        # 1) Exact / fuzzy check
        for (_, gold_norm, _) in self.gold_data:
            # Exact
            if cand_norm == gold_norm:
                return True
            # Fuzzy
            if fuzz.ratio(cand_norm, gold_norm) >= self.fuzzy_threshold:
                return True

        # 2) Semantic check (if it fails exact/fuzzy)
        if cand_emb is None:
            cand_emb = self.model.encode(cand_norm, convert_to_tensor=True)

        gold_stack = torch.stack([x[2] for x in self.gold_data])  # shape [N, dim]
        sim_scores = util.cos_sim(cand_emb, gold_stack)           # shape [1, N]
        max_score = torch.max(sim_scores).item()
        return (max_score >= self.semantic_threshold)


def build_labeled_examples(doc_data: list, max_ngram: int = 4,
                           fuzzy_threshold: int = 85, semantic_threshold: float = 0.75):
    """
    Given a list of documents (each with "text" and "keywords"),
    generate labeled examples (text, candidate, label).
    
    :param doc_data: List[{"id":..., "text":..., "keywords":...}]
    :param max_ngram: Maximum n-gram size for candidate generation
    :param fuzzy_threshold: Fuzzy ratio threshold for 'thefuzz' matching
    :param semantic_threshold: Cosine similarity threshold for embeddings
    :return: List of tuples (doc_text, candidate_string, label)
    """
    examples = []
    for doc in doc_data:
        text = doc["text"]
        gold_keywords = doc["keywords"]

        # Build a HybridMatcher once per doc
        hm = HybridMatcher(gold_keywords,
                           sbert_model,
                           fuzzy_threshold=fuzzy_threshold,
                           semantic_threshold=semantic_threshold)

        # Generate candidates
        candidates = generate_candidates(text, max_ngram=max_ngram)

        # Label each candidate
        for cand in candidates:
            label = 1 if hm.is_gold_candidate(cand) else 0
            examples.append((text, cand, label))

    return examples


##########################################
#  D) Main Preprocessing Pipeline Func   #
##########################################

def preprocess_data(translated_data: list):
    """
    Primary function to:
      1) Take already-translated data (list of dicts with "id","text","keywords").
      2) Split into train/dev/test.
      3) Build labeled examples for each split.
    
    :param translated_data: Output of translate_into_greek(), 
                           i.e. [{'id': '2', 'text': '...', 'keywords': [...]}, ...]
    :return: A dict containing the labeled examples for train/dev/test
             e.g. {
               "train": [...],
               "dev":   [...],
               "test":  [...]
             }
    """
    # 1) Split data into train/dev/test
    #    ~10% test, 10% dev, 80% train (adjust as needed)
    train_temp, test_data = train_test_split(translated_data, test_size=0.1, random_state=42)
    train_data, dev_data = train_test_split(train_temp, test_size=0.1111, random_state=42)

    print(f"Train docs: {len(train_data)} | Dev docs: {len(dev_data)} | Test docs: {len(test_data)}")

    # 2) Build labeled examples for each split
    train_examples = build_labeled_examples(train_data, max_ngram=4)
    dev_examples   = build_labeled_examples(dev_data,   max_ngram=4)
    test_examples  = build_labeled_examples(test_data,  max_ngram=4)

    print(f"Train examples: {len(train_examples)}")
    print(f"Dev examples:   {len(dev_examples)}")
    print(f"Test examples:  {len(test_examples)}")

    # Return the examples for downstream usage (model training, etc.)
    return {
        "train": train_examples,
        "dev":   dev_examples,
        "test":  test_examples
    }
