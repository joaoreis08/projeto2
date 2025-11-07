import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
from dotenv import load_dotenv
import os
from dataset import get_model_name,tokenized_train_test,bert_tokenizer,generate_dataset

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average='weighted')["f1"]}

def training_arguments():
    training_args = TrainingArguments(
        output_dir="./bert_sentiment",
        eval_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./bert_logs",
        logging_steps=50,
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none"
    )
    return training_args

def trainer(model,tokenized_train,tokenized_test,tokenizer,training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer

def train(trainer):
    trainer.train()

if __name__ == '__main__':
    compute_metrics()

    model = get_model_name()
    dataset = generate_dataset()
    tokenized_test, tokenized_train = tokenized_train_test(dataset)
    tokenizer = bert_tokenizer(model)
    training_args = training_arguments()
    trainer = trainer()
    train(trainer)