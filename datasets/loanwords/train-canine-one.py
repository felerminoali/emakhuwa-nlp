import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import datasets
import random
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
import os
sys.path.append('../')
root_folder = '../../'
import argparse

def run(args):

    # Set the device to GPU 1 (assuming it is available)
    device = torch.device("cuda:1")

    # Load the Canine tokenizer and BERT model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=root_folder+"checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=1,
    )

    # Load dataset
    data_files = {"train": "train.csv", "test": "test.csv"}
    print(args.data_path)
    dataset = load_dataset(args.data_path, data_files=data_files)

    train = dataset['train']
    test = dataset['test']

    # Split the training dataset into training and validation sets (80% training, 20% validation)
    split_ratio = 0.8
    train_size = int(len(train) * split_ratio)
    validation_dataset = train[train_size:]
    train_dataset = train[:train_size]

    train_dataset = train.shuffle(seed=42)
    validation_dataset = train.shuffle(seed=42)
    test_dataset = test.shuffle(seed=42)



    def preprocess_function(examples):
        """Preprocesses the data by tokenizing the text and truncating sequences to the maximum input length of the Canine tokenizer."""
        concatenated_text = examples["word"] 
        encodings = tokenizer(concatenated_text, padding='max_length', truncation=True,  return_tensors="pt")
        labels = examples["label"]
        return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "label": labels}


    # one
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    validation_dataset = validation_dataset.map(preprocess_function, batched=True)

	# Train the model
    trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=validation_dataset,
	)
    
    trainer.train()
    model.eval()
    model.save_pretrained(args.model_save)


    predictions = trainer.predict(test_dataset)

    # Get the predicted labels and convert them to numpy array
    predicted_labels = predictions.predictions.argmax(axis=1)

    # Get the true labels from the test dataset and convert them to numpy array
    true_labels = test_dataset["label"]

    data = {
        'word': test_dataset["word"],
        'tag':test_dataset["tag"],
        'label': true_labels,
        'prediction': predicted_labels
    }

    df = pd.DataFrame(data)
    filename_ = 'prediction-one-'+args.lang_pair+'.csv'
    df.to_csv(args.predictions_save_path+filename_, index=False)


    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    repot_txt = f"""
    f1-score :  {f1:.4f}
    precision :  {precision:.4f}
    recall :  {recall:.4f}
    accuracy :  {accuracy:.4f}
    """

    print(repot_txt)

if __name__ == '__main__':
    # Initialize the Parser
    parser = argparse.ArgumentParser(description = 'Generate translations and save them to json file.')
    
    lang = 'eng'
    transformer_model_name = 'google/canine-c'

    # Add arguments
    # Add arguments
    #parser.add_argument('-cmp','--checkpoint_model_path', type=str, metavar='', default=root_folder+'checkpoints/model-.pt', required=False, help='Model folder checkpoint path.')
    parser.add_argument('-psp','--predictions_save_path', type=str, metavar='', default=root_folder+"predictions/", required=False, help='Folder path to save predictions after inference.')
    parser.add_argument('-dp','--data_path', type=str, metavar='', default=root_folder+"data/", required=False, help='Test json path.')
    parser.add_argument('-ms','--model_save', type=str, metavar='', default=root_folder+"checkpoints/model_concat_"+lang+".pt", required=False, help='Test json path.')


    parser.add_argument('-mn','--model_name', type=str, metavar='', default=transformer_model_name, required=False, help='Model name.')
    parser.add_argument('-tn','--tokenizer_name', type=str, metavar='', default=transformer_model_name, required=False, help='Tokenizer name.')

    # Parse arguments
    args = parser.parse_args()

    print(args)

    # Start tokenization, encoding and generation
    run(args)