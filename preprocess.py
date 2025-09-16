# preprocess.py
import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
import streamlit as st
from typing import Tuple

# Cache the model so it only loads once
@st.cache_resource

#loads your classifier, base BERT model, tokenizer, and sets device (CPU/GPU).
def load_model(model_save_path: str):
    """
    Load fine-tuned classifier model, BERT embedder and tokenizer.
    Returns (classifier, bert_model, tokenizer, device)
    """
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"Model directory '{model_save_path}' not found.")
    model_classification = BertForSequenceClassification.from_pretrained(model_save_path)
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    model_bert = BertModel.from_pretrained(model_save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_classification.to(device).eval()
    model_bert.to(device).eval()
    return model_classification, model_bert, tokenizer, device

#chops DNA sequences into k-mers of length k
def sequence_to_kmers(sequence: str, k: int = 6) -> str:
    """
    Convert a DNA sequence string to k-mer tokens separated by spaces.
    """
    if not isinstance(sequence, str):
        return ""
    return ' '.join(sequence[i:i+k] for i in range(len(sequence) - k + 1)) if len(sequence) >= k else sequence
