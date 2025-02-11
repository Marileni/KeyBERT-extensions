**Deep Natural Language Processing Project**: Topic Modeling

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
