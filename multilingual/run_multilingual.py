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
    
    # 3) Fine-tune the model on train_examples/dev_examples
    # (You can tune any parameters, such as epochs, LR, output path, etc.)
    model = train_model(
        train_examples,
        dev_examples,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        output_path="my_bi_encoder_model",
        num_epochs=4,
        batch_size=8,
        warmup_steps=100,
        learning_rate=3e-5
    )
    
    # 4) Evaluate, output results, etc.
    
