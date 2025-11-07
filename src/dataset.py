import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
from dotenv import load_dotenv
import os

def get_device_name():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def get_model_name():
    load_dotenv()
    model_name = os.getenv('MODEL_NAME')
    return model_name

def generate_dataset():
    raw = load_dataset("imdb")
    train_small = raw['train'].select(range(1000))
    test_small = raw['test'].select(range(200))
    dataset = {"train": train_small, "test": test_small}
    print(dataset['train'][0])
    return dataset

def bert_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

def tokenized_train_test(dataset):
    tokenized_train = dataset['train'].map(tokenize_fn, batched=True)
    tokenized_test = dataset['test'].map(tokenize_fn, batched=True)

    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in ['input_ids','attention_mask','label']])
    tokenized_test = tokenized_test.remove_columns([c for c in tokenized_test.column_names if c not in ['input_ids','attention_mask','label']])

    tokenized_train.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    tokenized_test.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    print(tokenized_train[0])


if __name__ == '__main__':
    device = get_device_name()
    model_name = get_model_name()
    dataset = generate_dataset()
    tokenizer = bert_tokenizer(model_name)
    tokenize_fn()
    tokenized_train_test(dataset=dataset)