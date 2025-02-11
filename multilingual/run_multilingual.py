from .greek_translation import translate_into_greek
from .data_processing import preprocess_data
from .model_training import train_model

def run_multilingual():
    # 1) Translation into Greek
    translated_data = translate_into_greek()
    
    # 2) Preprocess the translated data (split, candidate generation, labeling, etc.)
    data_splits = preprocess_data(translated_data)
    train_examples = data_splits["train"]
    dev_examples   = data_splits["dev"]
    test_examples  = data_splits["test"]
    
    # 3) Train the model on train_examples
    model = train_model(train_examples, dev_examples)
    
    # 4) Evaluate, output results, etc.
    
