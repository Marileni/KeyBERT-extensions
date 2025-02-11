from .greek_translation import translate_into_greek
from .data_processing import preprocess_data
from .model_training import train_model

def run_domain():
    # 1) Translation into Greek
    translated_data = translate_into_greek()
    # 2) Load and preprocess data
    data = preprocess_data()
    # 3) Train or fine-tune your model
    model = train_model(data)
    # 4) Evaluate, output results, etc.
    
