import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
from dotenv import load_dotenv
import os
from dataset import get_model_name,get_device_name


def load_model():
    MODEL_NAME = get_model_name()
    device = get_device_name()
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    print(model.config)
    return model