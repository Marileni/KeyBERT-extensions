import random
import os
import unicodedata
import string

# For data splitting
from sklearn.model_selection import train_test_split

# For candidate generation
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModel, AutoTokenizer
from datasets import Dataset


from sentence_transformers import SentenceTransformer, models
from keybert import KeyBERT

from thefuzz import fuzz
import json


def process_data(file_path:str):
  # Convert dictionary values to a list
  data_list = list(data.values())  # Use values for the documents

  # Now perform the split on the list of documents
  train_temp, test_data = train_test_split(data_list, test_size=0.1, random_state=42)
  train_data, dev_data = train_test_split(train_temp, test_size=0.1111, random_state=42)

  print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

  # Save splits if desired
  with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
  with open("dev_data.json", "w", encoding="utf-8") as f:
      json.dump(dev_data, f, ensure_ascii=False, indent=2)
  with open("test_data.json", "w", encoding="utf-8") as f:
      json.dump(test_data, f, ensure_ascii=False, indent=2)

  return train_data, dev_data, test_data



def load_tokenizer_and_model():
    """Load pre-trained BERT tokenizer and model."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    return tokenizer, model

def tokenize_function(examples, tokenizer):
    """Tokenize dataset examples using the provided tokenizer."""
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128) 
