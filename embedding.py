# embedding.py
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

#Runs sequences through BERT and returns them as a NumPy array.
def compute_embeddings(sequences_list: List[str], bert_model, tokenizer, device, batch_size: int = 32) -> np.ndarray:
    """
    Compute CLS embeddings for sequences using BERT model in batches.
    Returns numpy array (n_sequences, hidden_size)
    """
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences_list), batch_size):
            batch_sequences = sequences_list[i:i+batch_size]
            encoded_inputs = tokenizer(
                batch_sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
            outputs = bert_model(**encoded_inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            if st.session_state.get('cancel_requested'):
                break
    if embeddings:
        return np.vstack(embeddings)
    hidden_size = getattr(getattr(bert_model, "config", None), "hidden_size", 768)
    return np.zeros((0, hidden_size))

#Runs sequences through the classifier model, returns the raw logits.
def predict_logits(classifier_model, tokenizer, device, sequences: List[str], batch_size: int = 32, temperature: float = 1.0):
    """
    Run classifier to get logits (CPU tensor). Applies temperature scaling if requested.
    """
    classifier_model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            encoded_inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
            outputs = classifier_model(**encoded_inputs)
            logits = outputs.logits
            if temperature != 1.0:
                logits = logits / float(temperature)
            preds.append(logits.cpu())
            if st.session_state.get('cancel_requested'):
                break
    if preds:
        return torch.cat(preds, dim=0)
    return torch.zeros((0, getattr(getattr(classifier_model, "config", None), "num_labels", 1)))
