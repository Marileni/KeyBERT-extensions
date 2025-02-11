from .greek_translation import translate_into_greek
from .data_processing import preprocess_data
from .model_training import train_model

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Utils for evaluation
from .utils.helper_functions import evaluate_keybert_fuzzy

def run_multilingual():
    # 1) Translate dataset to Greek
    translated_data = translate_into_greek()

    # 2) Preprocess: split into train/dev/test, generate labeled examples
    data_splits = preprocess_data(translated_data)
    train_examples = data_splits["train"]
    dev_examples   = data_splits["dev"]
    test_examples  = data_splits["test"]

    # 3) Train or fine-tune a Sentence-BERT model on (text, candidate, label) examples
    bi_model = train_model(
        train_examples,
        dev_examples,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        output_path="my_bi_encoder_model",
        num_epochs=4,
        batch_size=8,
        warmup_steps=100,
        learning_rate=3e-5
    )

    # 4) Load the fine-tuned model into KeyBERT
    fine_tuned_model = SentenceTransformer("./my_bi_encoder_model")
    kw_model_finetuned = KeyBERT(model=fine_tuned_model)

    # 5) Create a baseline KeyBERT model using original embeddings
    baseline_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    kw_model_baseline = KeyBERT(model=baseline_model)

    # 6) Evaluate on the test set (list of dicts with "text" and "keywords")
    p_ft, r_ft, f1_ft = evaluate_keybert_fuzzy(kw_model_finetuned, test_examples, top_n=5)
    print("=== Fine-tuned KeyBERT results (Test Set) ===")
    print(f"Precision: {p_ft:.4f}, Recall: {r_ft:.4f}, F1: {f1_ft:.4f}")

    p_base, r_base, f1_base = evaluate_keybert_fuzzy(kw_model_baseline, test_examples, top_n=5)
    print("\n=== Baseline KeyBERT results (Test Set) ===")
    print(f"Precision: {p_base:.4f}, Recall: {r_base:.4f}, F1: {f1_base:.4f}")

    print("\nMultilingual pipeline complete!")
