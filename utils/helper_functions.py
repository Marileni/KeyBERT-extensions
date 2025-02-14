from thefuzz import fuzz
import spacy

def precision_recall_f1_fuzzy(predicted_keywords, gold_keywords, threshold=80):
    """
    Compute precision, recall, and F1 using fuzzy matching.
    
    :param predicted_keywords: List of predicted keyword strings
    :param gold_keywords: List of gold keyword strings
    :param threshold: The fuzzy string matching threshold (0-100)
    :return: (precision, recall, f1) as floats
    """
    # Normalize
    gold_list = [g.lower().strip() for g in gold_keywords]
    pred_list = [p.lower().strip() for p in predicted_keywords]

    matched_preds = set()
    matched_gold = set()

    for p in pred_list:
        for g in gold_list:
            if fuzz.ratio(p, g) >= threshold:
                matched_preds.add(p)
                matched_gold.add(g)
                break

    tp = len(matched_preds)
    pred_count = len(pred_list)
    gold_count = len(gold_list)

    precision = tp / pred_count if pred_count else 0
    recall    = tp / gold_count if gold_count else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def evaluate_keybert_fuzzy(kw_model, docs, top_n=5, threshold=80):
    """
    Evaluate KeyBERT (or a KeyBERT-like model) on a set of documents,
    comparing extracted keywords to gold references using fuzzy matching.

    :param kw_model: A KeyBERT instance (or similar) with an .extract_keywords() method
    :param docs: A list of dicts, each with "text" and "keywords"
    :param top_n: Number of top keywords to extract
    :param threshold: Fuzzy match threshold
    :return: A tuple (avg_precision, avg_recall, avg_f1)
    """
    all_p, all_r, all_f1 = [], [], []

    for doc in docs:
        text = doc["text"]
        gold = doc["keywords"]

        predicted = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1,4),
            top_n=top_n
        )
        # predicted is typically a list of (keyword, score) pairs
        pred_kws = [kw for kw, score in predicted]

        p, r, f1 = precision_recall_f1_fuzzy(pred_kws, gold, threshold)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)

    avg_p = sum(all_p) / len(all_p)
    avg_r = sum(all_r) / len(all_r)
    avg_f1 = sum(all_f1) / len(all_f1)
    return avg_p, avg_r, avg_f1 

def evaluate_predictions_domain_adaptation(predicted_keywords_list, test_dataset, threshold=50):
    all_p, all_r, all_f1 = [], [], []

    for pred_kws, doc in zip(predicted_keywords_list, test_dataset):
        gold_kws = doc["keywords"]
        p, r, f1 = precision_recall_f1_fuzzy(pred_kws, gold_kws, threshold)

        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)

    avg_p = sum(all_p) / len(all_p)
    avg_r = sum(all_r) / len(all_r)
    avg_f1 = sum(all_f1) / len(all_f1)
    return avg_p, avg_r, avg_f1
def normalize_keywords(keyword_list, nlp):
    """
    Normalizes keywords by lowercasing, lemmatizing, and removing punctuation.

    Args:
        keyword_list (list): List of extracted keywords.
        nlp: spaCy NLP model.

    Returns:
        set: Normalized keyword set.
    """
    normalized_set = set()
    for kw in keyword_list:
        doc = nlp(kw.lower().strip())  
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]  
        normalized_set.add(" ".join(tokens))  

    return normalized_set

def evaluate_keywords_met(true_keywords, predicted_keywords):
    """
    Computes precision, recall, and F1-score for keyword extraction.

    Args:
        true_keywords (list): Reference (gold-standard) keywords.
        predicted_keywords (list): Extracted keywords.

    Returns:
        tuple: (precision, recall, f1-score)
    """
    nlp = spacy.load("en_core_web_sm")
    
    true_set = normalize_keywords(true_keywords, nlp)
    pred_set = normalize_keywords(predicted_keywords, nlp)

    tp = len(true_set & pred_set)  
    fp = len(pred_set - true_set)  
    fn = len(true_set - pred_set)  

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
    
def evaluate_results(df):
    """
    Evaluates the extracted keywords against the gold-standard keywords using precision, recall, and F1-score.
    
    Args:
        df (pd.DataFrame): The dataset containing gold-standard and extracted keywords.
    
    Returns:
        pd.DataFrame: The dataset with added evaluation scores for both original and filtered texts.
    """
        
    df["original_scores"] = df.apply(lambda row: evaluate_keywords_met(row["keywords"], row["original_keywords"]), axis=1)
    print("\nKeywords of original text evaluated!")
    df["filtered_scores"] = df.apply(lambda row: evaluate_keywords_met(row["keywords"], row["filtered_keywords"]), axis=1)
    print("\n!Keywords of preprocessed text evaluated!")
    
    return df
