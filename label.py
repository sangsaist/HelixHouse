# label.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict

#Finds index → label mapping
def load_label_mapping(model_clf, model_dir: str):
    """
    Try to obtain index->label map from model config or common files in model dir.
    Returns dict[int -> str] (empty if not found).
    """
    label_to_species = {}
    #model config id2label
    try:
        cfg = getattr(model_clf, "config", None)
        if cfg and hasattr(cfg, "id2label") and cfg.id2label:
            label_to_species = {int(k): v for k, v in cfg.id2label.items()}
            st.info("Loaded label map from model config (id2label).")
            return label_to_species
    except Exception:
        pass

    #try files in model dir
    candidate_files = ["label_map.json", "id2label.json", "index_to_label.json"]
    for fn in candidate_files:
        p = os.path.join(model_dir, fn)
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    j = json.load(fh)
                if isinstance(j, dict):
                    label_to_species = {int(k): v for k, v in j.items()}
                elif isinstance(j, list):
                    label_to_species = {i: j[i] for i in range(len(j))}
                st.info(f"Loaded label map from {fn}.")
                return label_to_species
            except Exception:
                continue
    return {}

#calculates Shannon and Simpson.
def calculate_diversity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Shannon and Simpson index per Sample_ID.
    Returns DataFrame indexed by Sample_ID with columns Shannon_Index and Simpson_Index.
    """
    def shannon_index(counts):
        p = counts / counts.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    def simpson_index(counts):
        p = counts / counts.sum()
        return np.sum(p ** 2)
    abundance_by_location = df.groupby(['Sample_ID', 'Taxonomy'])['Read_Count'].sum().unstack(fill_value=0)
    diversity_metrics = pd.DataFrame({
        'Shannon_Index': abundance_by_location.apply(shannon_index, axis=1),
        'Simpson_Index': abundance_by_location.apply(simpson_index, axis=1),
    })
    return diversity_metrics
