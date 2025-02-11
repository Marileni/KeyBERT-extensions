import os
import zipfile
import requests
import json
from transformers import MarianMTModel, MarianTokenizer

def download_and_extract_zip(url, zip_file, extract_to):
    """
    Download a ZIP file from a given URL and extract its contents.
    
    :param url: URL of the file to download
    :param zip_file: Local filename for the downloaded ZIP
    :param extract_to: Directory where the ZIP should be extracted
    """
    print(f"Downloading dataset from {url} ...")
    response = requests.get(url)
    with open(zip_file, "wb") as file:
        file.write(response.content)

    print(f"Extracting {zip_file} into {extract_to} ...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Download and extraction complete.")


def translate_text(text, model, tokenizer):
    """
    Translate a single piece of text using a MarianMT model.
    
    :param text: The string to translate
    :param model: The MarianMTModel for translation
    :param tokenizer: The MarianTokenizer corresponding to the model
    :return: Translated text (string)
    """
    try:
        # Tokenize and encode the input text
        inputs = tokenizer.encode(text, return_tensors="pt", 
                                 padding=True, truncation=True)
        # Generate translation
        translated_tokens = model.generate(inputs)
        # Decode translated tokens back to a string
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        # If an error occurs, return the original text instead of crashing
        return text


def process_dataset(docs_path, keys_path, output_file, model, tokenizer):
    """
    Process documents and keywords, translate them, and save results to a JSON file.
    
    :param docs_path: Directory containing .txt document files (e.g., "1.txt", "2.txt")
    :param keys_path: Directory containing corresponding .key files
    :param output_file: JSON file to store translated outputs
    :param model: MarianMTModel for translation
    :param tokenizer: MarianTokenizer for translation
    :return: A list of dictionaries with "id", "text", and "keywords"
    """
    output = []

    # Get all .txt files and sort them by numeric ID
    doc_files = sorted(os.listdir(docs_path), key=lambda x: int(x.split(".")[0]))

    for doc_file in doc_files:
        if doc_file.endswith(".txt"):
            doc_id = doc_file.split(".")[0]  # e.g., "2" from "2.txt"
            print(f"Processing document ID: {doc_id}")

            # Read document content
            doc_path = os.path.join(docs_path, doc_file)
            with open(doc_path, "r", encoding="utf-8") as f:
                document_text = f.read()

            # Normalize text to a single line
            document_text_single_line = " ".join(document_text.splitlines())

            # Translate document content to Greek
            translated_document = translate_text(document_text_single_line, model, tokenizer)

            # Find the corresponding .key file
            key_file = f"{doc_id}.key"
            key_path = os.path.join(keys_path, key_file)

            if os.path.exists(key_path):
                # Read keywords
                with open(key_path, "r", encoding="utf-8") as f:
                    keywords = [line.strip() for line in f]

                # Translate each keyword to Greek
                translated_keywords = [translate_text(kw, model, tokenizer) for kw in keywords]

                # Clean up any problematic characters from the translated text
                for char in ["\n", "\t", '"', "'", ";"]:
                    translated_document = translated_document.replace(char, " ")

                print(f"Translated document: {translated_document[:60]}...")  # Preview
                print(f"Translated keywords: {translated_keywords}")

                # Append the translated data to the output list
                output.append({
                    "id": doc_id,
                    "text": translated_document,
                    "keywords": translated_keywords
                })

    # Save the translated data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Translation complete. Output saved to {output_file}")
    return output


def translate_into_greek():
    """
    Main function to:
      1) Download and extract the Inspec dataset
      2) Initialize the MarianMT model for English->Greek
      3) Process and translate the dataset
      4) Return the translated dataset as a Python list
    
    :return: List of dicts where each dict has 'id', 'text', and 'keywords' in Greek
    """
    # URLs and Paths
    dataset_url = "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip"
    zip_file = "Inspec.zip"
    extract_to = "Inspec"

    docs_path = os.path.join(extract_to, "Inspec", "docsutf8")
    keys_path = os.path.join(extract_to, "Inspec", "keys")
    output_file = "Inspec_translated.json"

    # Step 1) Download and Extract the Dataset
    if not os.path.exists(extract_to):
        download_and_extract_zip(dataset_url, zip_file, extract_to)
    else:
        print(f"{extract_to} folder already exists. Skipping download and extraction.")

    # Step 2) Initialize the MarianMT Translation Model (English -> Greek)
    model_name = "Helsinki-NLP/opus-mt-en-el"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Step 3) Process the dataset to translate documents and keywords into Greek
    translated_data = process_dataset(docs_path, keys_path, output_file, model, tokenizer)

    return translated_data
