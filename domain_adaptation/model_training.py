# For data splitting
from sklearn.model_selection import train_test_split
import spacy 


def model_training(dataset, tokenized_train_dataset, tokenized_val_dataset, data_collator, model):
  # Split dataset into train (80%), validation (10%), and test (10%)
  train_val_test_split = dataset.train_test_split(test_size=0.2, seed=42)  # First, split 80%-20%
  val_test_split = train_val_test_split["test"].train_test_split(test_size=0.5, seed=42)  # Split the 20% into 10%-10%

 # Assign splits
  train_dataset = train_val_test_split["train"]  # 80%
  val_dataset = val_test_split["train"]  # 10%
  test_dataset = val_test_split["test"]  # 10%


  # Training arguments
  training_args = TrainingArguments(
      output_dir="./fine_tuned_bert",
      evaluation_strategy="epoch",
      logging_strategy="epoch",  # Registra la loss ogni epoca
      learning_rate=5e-5,
      per_device_train_batch_size=8,
      num_train_epochs=3,
      save_steps=500
  )

  # Trainer setup
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train_dataset,
      eval_dataset=tokenized_val_dataset,
      data_collator=data_collator,
  )

  # Start training
  trainer.train()
 
  return trainer 
