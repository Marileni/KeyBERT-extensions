import os
import zipfile
import requests
import pandas as pd
import re
import spacy
from transformers import pipeline

    
def preprocess_data():
    """
    Downloads, extracts, and processes the Krapivin2009 dataset into a structured Pandas DataFrame.
    
    Returns:
        df (pd.DataFrame): A DataFrame containing document filenames, raw text, and gold-standard keywords.
    """
    
    # Define dataset URL and file paths
    dataset_url = "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Krapivin2009.zip"
    zip_file = "Krapivin2009.zip"
    extract_to = "Krapivin2009"
    

    # Step 1: Download the dataset from the given URL
    print("Downloading dataset...")
    response = requests.get(dataset_url)
    with open(zip_file, "wb") as f:
        f.write(response.content)

    # Step 2: Extract the ZIP file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        
    # Define document and keyword directories
    docs_path = os.path.join(extract_to, "docsutf8")  # Folder containing .txt files
    keys_path = os.path.join(extract_to, "keys")  # Folder containing .keys files

    dataset = {}    # Dictionary to store processed data

    # Step 3: Read documents and keywords
    print("Loading documents and keywords into DataFrame...")    
    for file in os.listdir(docs_path):  # Ensure matching order
        if file.endswith(".txt"):
            file_path = os.path.join(docs_path, file)
            key_path = os.path.join(keys_path, file.replace(".txt", ".keys"))  # Corresponding keys file
        
            # Read document text
            with open(file_path, "r", encoding="utf-8") as f:
                document = f.read().strip()
                
            # Read corresponding keywords
            keywords = []
            if os.path.exists(key_path):    # Corrected from keys_path to key_path
                with open(key_path, "r", encoding="utf-8") as f:
                    keywords = [line.strip() for line in f.readlines()]

            dataset[file] = {"text": document, "keywords": keywords}

    # Convert dictionary to DataFrame
    df = pd.DataFrame([
        {"filename": k, "text": v["text"], "keywords": v["keywords"]}
        for k, v in dataset.items()
    ])
    
    print("Dataframe successfully loaded!")
    return df

def filter_named_entities(text, ner_pipeline):
    """
    Removes named entities (PERSON, ORGANIZATION, LOCATION, MISC) from the text.

    Args:
        text (str): The input text.
        ner_pipeline: Hugging Face NER pipeline model.

    Returns:
        str: The text with named entities removed.
    """    
        
    # Identify named entities
    entities = ner_pipeline(text, aggregation_strategy="simple")

    # Extract named entities to remove
    to_remove = {ent["word"] for ent in entities if ent["entity_group"] in ["PER", "ORG", "LOC", "MISC"]}

    # Replace named entities with an empty string, ensuring sentence structure remains intact
    filtered_text = text
    for entity in to_remove:
        filtered_text = re.sub(r'\b' + re.escape(entity) + r'\b', '', filtered_text, flags=re.IGNORECASE)

    return filtered_text

def ner_filter(df):
    """
    Applies named entity removal to all documents in the dataset.

    Args:
        df (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The dataset with an additional column containing the filtered text.
    """
    print("Applying Named Entity Recognition (NER) filtering...")
    
    # Load the NER model only once for efficiency
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    df["filtered_text"] = df["text"].apply(lambda x: filter_named_entities(x, ner_pipeline))
    return df

def extract_keywords(text, kw_model, top_n=8):
    """
    Extracts keywords from a given text using KeyBERT.

    Args:
        text (str): The input text.
        kw_model: A KeyBERT model instance.
        top_n (int): Number of keywords to extract.

    Returns:
        list: Extracted keywords.
    """
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words="english", top_n=top_n)
    return [kw[0] for kw in keywords]  # Extract only keyword strings

def apply_kb(df, kw_model):
    """
    Applies KeyBERT to extract keywords from both original and filtered texts.

    Args:
        df (pd.DataFrame): The dataset.
        kw_model: A KeyBERT model instance.

    Returns:
        pd.DataFrame: The dataset with additional keyword columns.
    """
    print("Extracting keywords using KeyBERT...")
    
    df["original_keywords"] = df["text"].apply(lambda x: extract_keywords(x,kw_model, top_n=8))
    print("\nKeywords from original text extracted!")
    df["filtered_keywords"] = df["filtered_text"].apply(lambda x: extract_keywords(x, kw_model, top_n=8))
    print("\nKeywords from preprocessed text extracted!")
    
    return df
