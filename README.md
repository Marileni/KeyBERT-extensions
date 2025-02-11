# Deep Natural Language Processing Project: Topic Modeling

This repository focuses on **topic modeling** techniques that leverage BERT-based keyword extraction. We explore three main approaches:

1. **Domain Adaptation** – Applying keyword extraction in a specialized domain (e.g., agriculture).  
2. **Multilingual Extension** – Handling documents in the Greek language.
3. **NER-based Preprocessing** – Using Named Entity Recognition to filter key entities before extracting keywords.

## Repository Structure

- `domain_adaptation/` covers the agriculture-domain adaptation approach.
- `multilingual/` includes all code for multilingual (Greek) modeling.
- `ner_preprocessing/` implements NER-based entity filtering.
- `utils/` has utility scripts for logging, helper functions, etc.

## How to Run

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   

2.  **Running via `main.py`**

      We provide a single **entry point** in `main.py` that accepts a parameter specifying which approach to run:
   
      ```bash
      python main.py --approach domain
      ```
      Runs the domain adaptation pipeline.
      
      ```bash
      python main.py --approach multilingual
      ```
      Runs the multilingual (Greek) pipeline.
      
      ```bash
      python main.py --approach ner
      ```
      Runs the NER-based preprocessing pipeline.
      
      Inside `main.py`, these commands map to the corresponding scripts in their respective folders.

   
      ### Alternatively, you can run each approach directly:
      
       **Domain Adaptation:**
      ```bash
      python domain_adaptation/run_domain_adaptation.py
      ```
      
      **Multilingual (Greek):**
      ```bash
      python multilingual/run_multilingual.py
      ```
      
      **NER Preprocessing:**
      ```bash
      python ner_preprocessing/run_ner_preprocessing.py
      ```
      Each script calls the relevant modules (`data_processing.py` and `model_training.py`) to complete its tasks.


## Results
During each run, the code may generate:
- **Logs**: Training and validation logs for model performance tracking.
- **Metrics**: Precision, Recall, F1 scores for keyword extraction.
- **Comparison**: We compare the final results (baseline vs. extended approaches) in our final report.


