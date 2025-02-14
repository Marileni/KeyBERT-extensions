from .data_processing import preprocess_data, ner_filter, apply_kb, evaluate_results
from keybert import KeyBERT
import numpy as np

def run_ner():
    """
    Runs the full Named Entity Recognition (NER) preprocessing and KeyBERT keyword extraction pipeline.
    
    Steps:
    1. Loads and preprocesses the dataset.
    2. Applies Named Entity Recognition (NER) filtering.
    3. Extracts keywords using KeyBERT for both original and filtered text.
    4. Evaluates extracted keywords against gold-standard keywords.
    5. Computes global precision, recall, and F1-score.
    
    Prints the evaluation results.
    """
    
    # 1) Load data into a Dataframe
    df = preprocess_data()
    
    # Step 2: Apply Named Entity Recognition (NER) filtering
    df = ner_filter(df)
    
    # Step 3: Load KeyBERT model and extract keywords
    kw_model = KeyBERT()
    df = apply_kb(df, kw_model)
    
    # Step 4: Evaluate extracted keywords against gold-standard keywords
    df = evaluate_results(df)
    
    # Step 5: Compute global precision, recall, and F1-score
    original_metrics = np.mean(np.vstack(df["original_scores"]), axis=0)
    filtered_metrics = np.mean(np.vstack(df["filtered_scores"]), axis=0)

    # Print evaluation results
    print("\n=== KeyBERT results (Original text) ===")
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(*original_metrics))    
    print("\n=== KeyBERT results (Filtered text) ===")
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(*filtered_metrics))
    
    print("\nNer preprocessing pipeline complete!")
    
