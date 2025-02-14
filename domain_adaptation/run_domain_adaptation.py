    
from .data_processing import preprocess_data
from .model_training import train_model
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
# Utils for evaluation
from .utils.helper_functions import evaluate_predictions_domain_adaptation
from .structured_dataset import json_creation

def run_domain_adaptation():
  data= json_creation()
  
  
  
  train_data, val_data, test_data = process_data(data)
  texts = [entry["text"] for entry in data.values()]
  
  tokenizer, model = load_tokenizer_and_model()
    
  tokenized_output = tokenize_function(texts, tokenizer)
  dataset = Dataset.from_dict({"text": texts})

  # Tokenize datasets
  tokenized_train_dataset = train_data.map(tokenized_output, batched=True)
  tokenized_val_dataset = val_data.map(tokenized_output, batched=True)
  tokenized_test_dataset=test_data.map(tokenized_output, batched=True)

  # Data collator for MLM
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

  trainer=model_training(dataset, tokenized_train_dataset, tokenized_val_dataset, data_collator, model)

  output_directory = "./fine_tuned_bert"
  
  model.save_pretrained(output_directory)
  tokenizer.save_pretrained(output_directory)

  # Upload il tuo BERT fine-tuned
  bert_model = AutoModel.from_pretrained(output_directory)
  tokenizer = AutoTokenizer.from_pretrained(output_directory)

  # we create the SentenceTransformer with our model BERT fine-tuned
  word_embedding_model = models.Transformer(output_directory, max_seq_length=256)
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

  # we combine the models to obtain a SentenceTransformer
  fine_tuned_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

  # now we can use with KeyBERT
  kw_model_finetuned = KeyBERT(model=fine_tuned_model)

  kw_model_base = KeyBERT()  # KeyBERT with BERT-base

  keywords_base = kw_model_base.extract_keywords(test_data, keyphrase_ngram_range=(1, 2), stop_words='english',top_n=5)
  keywords_finetuned = kw_model_finetuned.extract_keywords(test_data, keyphrase_ngram_range=(1, 2), stop_words='english',top_n=5)

  test_datanew = [{"text": doc, "keywords": data[doc_id]["keywords"]}
                 for doc_id, doc in zip(data.keys(), test_data)]


  predicted_keywords_baseline = keywords_base  # List of predicted keywords from baseline KeyBERT
  predicted_keywords_finetuned = keywords_finetuned  # List of predicted keywords from fine-tuned KeyBERT 

  p_base, r_base, f1_base = evaluate_predictions_domain_adaptation(predicted_keywords_baseline, test_datanew, threshold = 50)
  print("\nBaseline KeyBERT Results:")
  print(f"Precision: {p_base:.4f}, Recall: {r_base:.4f}, F1: {f1_base:.4f}")

  p_ft, r_ft, f1_ft = evaluate_predictions_domain_adaptation(predicted_keywords_finetuned, test_datanew, threshold= 50)
  print("\nFine-Tuned KeyBERT Results:")
  print(f"Precision: {p_ft:.4f}, Recall: {r_ft:.4f}, F1: {f1_ft:.4f}")

  print("\nDomain adaptation pipeline complete!")

  

  
