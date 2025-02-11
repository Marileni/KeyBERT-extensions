import os
from typing import List, Tuple

# Sentence-Transformers training
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from torch.utils.data import DataLoader
import torch

########################################
#  Build Bi-Encoder Input Examples     #
########################################

def build_bi_encoder_examples(labeled_data: List[Tuple[str, str, int]]):
    """
    Convert labeled_data (list of (text, candidate, label)) 
    into InputExample objects for Sentence-Transformers.
    
    :param labeled_data: A list of tuples (doc_text, candidate, label).
    :return: A list of Sentence-Transformers InputExample objects.
    """
    st_examples = []
    for (doc_text, candidate, lab) in labeled_data:
        # Convert label to float for BinaryClassificationEvaluator
        label_float = float(lab)
        example = InputExample(texts=[doc_text, candidate], label=label_float)
        st_examples.append(example)
    return st_examples


##############################################
#  Main Training Function for Bi-Encoder     #
##############################################

def train_model(
    train_examples: List[Tuple[str, str, int]],
    dev_examples: List[Tuple[str, str, int]],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    output_path: str = "my_bi_encoder_model",
    num_epochs: int = 4,
    batch_size: int = 8,
    warmup_steps: int = 100,
    learning_rate: float = 3e-5
):
    """
    Fine-tunes a Sentence-BERT bi-encoder model on labeled data.
    Each labeled data point is (text, candidate, label), 
    indicating whether 'candidate' is a valid keyword for 'text'.
    
    :param train_examples: Labeled training data [(text, candidate, label), ...]
    :param dev_examples:   Labeled development (validation) data
    :param model_name:     Pretrained Sentence-Transformers model to start with
    :param output_path:    Where the fine-tuned model will be saved
    :param num_epochs:     Number of epochs to train
    :param batch_size:     Training batch size
    :param warmup_steps:   Steps for learning rate warm-up
    :param learning_rate:  Learning rate for AdamW optimizer
    
    :return: A fine-tuned SentenceTransformer model
    """

    # 1. Convert raw (text, candidate, label) data into InputExample objects
    train_st_ex = build_bi_encoder_examples(train_examples)
    dev_st_ex   = build_bi_encoder_examples(dev_examples)

    print(f"Bi-Encoder Train Examples: {len(train_st_ex)} | Dev Examples: {len(dev_st_ex)}")

    # 2. Load the base SentenceTransformer model
    bi_model = SentenceTransformer(model_name)
    print(f"Loaded base model: {model_name}")

    # 3. Prepare DataLoader and Loss Function
    train_dataloader = DataLoader(train_st_ex, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=bi_model)

    # 4. Prepare an evaluator on the dev set (BinaryClassificationEvaluator expects InputExample format)
    dev_evaluator = BinaryClassificationEvaluator.from_input_examples(
        dev_st_ex,
        batch_size=batch_size,
        name="dev_eval"
    )

    # 5. Fine-tune the model
    bi_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        evaluation_steps=len(train_dataloader),
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )

    # 6. Save the fine-tuned model
    bi_model.save(output_path)
    print(f"Model fine-tuned and saved to: {output_path}")

    return bi_model 
