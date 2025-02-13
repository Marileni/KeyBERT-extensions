from thefuzz import fuzz

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
