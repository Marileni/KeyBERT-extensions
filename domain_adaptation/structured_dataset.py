import zipfile
import requests 
import os 
import json

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


def process_dataset(docs_path, keys_path, output_file):
    """
    Process documents and keywords, translate them, and save results to a JSON file.
    
    :param docs_path: Directory containing .txt document files (e.g., "1.txt", "2.txt")
    :param keys_path: Directory containing corresponding .key files
    :param output_file: JSON file to store translated outputs
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

            

            # Find the corresponding .key file
            key_file = f"{doc_id}.key"
            key_path = os.path.join(keys_path, key_file)

            if os.path.exists(key_path):
                # Read keywords
                with open(key_path, "r", encoding="utf-8") as f:
                    keywords = [line.strip() for line in f]

              


                # Append the translated data to the output list
                output.append({
                    "id": doc_id,
                    "text": document_text_single_line,
                    "keywords": keywords
                })

    # Save the translated data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Output saved to {output_file}")
  
    return output

def json_creation():
  # URLs and Paths
    dataset_url = "https://github.com/LIAAD/KeywordExtractor-Datasets/blob/master/datasets/fao780.zip"
    zip_file = "fao780.zip"
    extract_to = "fao780"

    docs_path = os.path.join(extract_to, "fao780", "docsutf8")
    keys_path = os.path.join(extract_to, "fao780", "keys")
    output_file = "fao780_dataset_renamed.json"

        # Step 1) Download and Extract the Dataset
    if not os.path.exists(extract_to):
        download_and_extract_zip(dataset_url, zip_file, extract_to)
    else:
        print(f"{extract_to} folder already exists. Skipping download and extraction.")

    data=process_dataset(docs_path, keys_path, output_file)

    return data
