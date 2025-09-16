# app.py — merged frontend (UI + viz) and orchestrator
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from umap import UMAP
import hdbscan

# import backend modules (keep these as separate files)
from preprocess import load_model, sequence_to_kmers
from embedding import compute_embeddings, predict_logits
from label import load_label_mapping, calculate_diversity_metrics

# -------------------------
# Page config + CSS (exact look preserved)
# -------------------------
st.set_page_config(
    page_title="eDNA — Biodiversity Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Global */
      .main { padding: 2.2rem 2.2rem 1.5rem; }
      section[data-testid="stSidebar"] { width: 360px !important; }
      [data-testid="stSidebarNav"] { display:none; }
      .st-emotion-cache-1y4p8pa { padding-top: 1rem; } /* reduce top padding inside tabs */

      /* Headline gradient */
      .app-title {
        font-weight: 800; letter-spacing: 0.3px;
        background: linear-gradient(90deg, #0ea5e9 0%, #7c3aed 90%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
      }
      .app-subtitle { color: var(--text-color, #5b5b6b); opacity: 0.85; }

      /* Info banner */
      .banner {
        background: linear-gradient(90deg, rgba(14,165,233,.12), rgba(124,58,237,.12));
        border: 1px solid rgba(125,125,125,.15);
        padding: 12px 16px; border-radius: 14px; margin: 8px 0 18px;
      }

      /* Card metrics */
      .kpi {
        border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(0,0,0,.07);
        background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(250,250,252,.85));
        box-shadow: 0 8px 22px rgba(0,0,0,.04);
      }
      .kpi .label { font-size: 13.5px; color: #6b7280; }
      .kpi .value { font-size: 24px; font-weight: 800; margin-top: 6px; }

      /* Section header */
      .section-title {
        font-weight: 700; font-size: 18px; margin: 10px 0 6px;
      }
      .divider {
        height: 1px; background: linear-gradient(to right, rgba(0,0,0,.08), rgba(0,0,0,0));
        margin: 10px 0 14px;
      }

      /* Dataframes */
      .stDataFrame { border-radius: 12px; overflow: hidden; }

      /* Buttons */
      .stButton > button {
        border-radius: 12px !important; font-weight: 700 !important;
        padding: 0.6rem 1rem !important;
      }

      /* Progress & statuses */
      .status-pill {
        display:inline-flex; gap:8px; align-items:center;
        padding:6px 10px; border-radius:999px;
        border:1px solid rgba(0,0,0,.08);
        background: rgba(14,165,233,.08);
      }
      .status-pill .dot {
        width:8px; height:8px; border-radius:999px; background:#0ea5e9;
      }

      /* Help text small */
      .muted { color: #6b7280; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Sidebar controls (identical to your previous UI)
# -------------------------
st.sidebar.title("🧪 Controls")
st.sidebar.caption("Upload your `edna.csv`, set model directory and run analysis.")

with st.sidebar:
    with st.expander("📁 Data Source", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
        use_default_path = st.checkbox("Use default local CSV path", value=True)
        default_path = "C:/Users/sanja/Desktop/SIH/Prototype/edna.csv"
        csv_path_input = st.text_input("Or enter CSV path", value=(default_path if use_default_path else ""))

    with st.expander("🧠 Model & Inference", expanded=True):
        model_dir = st.text_input("Model directory", value="./fine_tuned_model", help="Folder with config.json, pytorch_model.bin, tokenizer files, and optional label_map.json")
        batch_size = st.selectbox("Batch size (inference / embeddings)", options=[8,16,32,64], index=2)
        conf_threshold = st.slider("Prediction confidence threshold", 0.0, 1.0, 0.55, 0.01,
                                   help="If top-1 probability < threshold we'll mark prediction as Uncertain.")
        use_temp = st.checkbox("Use temperature scaling (manual)", value=False)
        temp_value = st.slider("Temperature", 0.1, 5.0, 1.0, 0.1) if use_temp else 1.0

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_button = st.button("🚀 Load model & Run analysis", type="primary", use_container_width=True)
    with col_btn2:
        cancel_button = st.button("✋ Cancel", use_container_width=True)

    st.markdown("---")
    st.caption("Tip: Provide a `label_map.json` matching training labels for perfect index→label mapping.")

# -------------------------
# Header / Banner
# -------------------------
st.markdown('<h1 class="app-title">eDNA — AI-Driven Biodiversity Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Predict taxonomy, compute diversity metrics, and discover novel taxa from eDNA.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="banner">
      <span class="status-pill"><span class="dot"></span> Ready</span>
      <span style="margin-left:10px" class="muted">Upload data and click <b>Run analysis</b>. Uses BERT embeddings, top-3 predictions, Shannon/Simpson, HDBSCAN, and UMAP.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How it works (quick)"):
    st.write(
        """
        **Pipeline**
        1) Convert sequences to **k-mers**, compute **BERT embeddings**, and predict taxonomy for unassigned sequences (top-3 with confidences).  
        2) Compute **diversity metrics**: Shannon (richness+evenness) and Simpson (dominance).  
        3) For predicted items, run **HDBSCAN** to group potential **novel taxa** and project with **UMAP** for visualization.
        """
    )

# -------------------------
# Status placeholders + session state
# -------------------------
status_col1, status_col2, status_col3 = st.columns([1,1,2])
status_msg = status_col1.empty()
progress_placeholder = status_col2.progress(0)
device_placeholder = status_col3.empty()

if 'cancel_requested' not in st.session_state:
    st.session_state.cancel_requested = False

if cancel_button:
    st.session_state.cancel_requested = True
    st.warning("Cancellation requested. Long-running operations will stop at the next checkpoint.")

# -------------------------
# Plot helpers (moved from viz.py into this single app frontend)
# -------------------------
import plotly.express as px

def plot_abundance_bar(abundance_df: pd.DataFrame, selected_sample: str, top_k: int = 15):
    fig = px.bar(
        abundance_df.head(top_k),
        x='Taxonomy',
        y='Read_Count',
        title=f"Top {top_k} Taxa by Read Count — {selected_sample}",
        labels={'Read_Count': 'Read Count', 'Taxonomy': 'Taxonomy'},
        color='Read_Count',
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_tickangle=-40, showlegend=False, template="plotly_white", margin=dict(l=10,r=10,t=60,b=10))
    return fig

def plot_novel_umap(novel_taxa_in_sample: pd.DataFrame, selected_sample: str):
    tooltip_cols = ['Taxonomy', 'Read_Count', 'Sequence']
    novelty_fig = px.scatter(
        novel_taxa_in_sample,
        x='umap_x', y='umap_y',
        color='Taxonomy',
        hover_data=tooltip_cols,
        title=f'UMAP — novel taxa in {selected_sample}',
    )
    novelty_fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=60,b=10))
    return novelty_fig

# -------------------------
# Main run logic (keeps exact process)
# -------------------------
if run_button:
    st.session_state.cancel_requested = False
    start_time = time.time()

    # 1) Load model
    try:
        with st.spinner("Loading model & tokenizer..."):
            model_clf, model_emb, tokenizer, device = load_model(model_dir)
        status_msg.success("Model loaded.")
        device_placeholder.info(f"🖥️ Inference device: **{device}**")
    except Exception as e:
        st.error("Failed to load model/tokenizer.")
        st.exception(e)
        st.stop()

    # try to load label map (from model)
    label_to_species = load_label_mapping(model_clf, model_dir)
    species_to_label = {v: k for k, v in label_to_species.items()}

    # validate label map length vs model.num_labels if available
    model_num_labels = getattr(getattr(model_clf, "config", None), "num_labels", None)
    if label_to_species and model_num_labels is not None:
        if len(label_to_species) != model_num_labels:
            st.error(f"Label map length ({len(label_to_species)}) does not match model.num_labels ({model_num_labels}). Add a correct label_map.json.")
            st.stop()

    # Model & Data status panel (compact)
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
        c1.metric("Model dir", model_dir if len(model_dir) <= 22 else "…" + model_dir[-22:])
        try:
            model_files = os.listdir(model_dir)
            c2.metric("# model files", f"{len(model_files)}")
        except Exception:
            c2.metric("# model files", "—")
        c3.metric("Device", str(device))
        c4.metric("Run started", time.strftime("%H:%M:%S", time.localtime(start_time)))

    # 2) Load CSV data (uploaded or path)
    try:
        status_msg.info("Loading CSV…")
        with st.spinner("Reading CSV…"):
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                if not csv_path_input:
                    st.error("No CSV path provided. Upload a file or provide a path.")
                    st.stop()
                df = pd.read_csv(csv_path_input)
        status_msg.success(f"Data loaded — {len(df)} rows.")
    except Exception as e:
        st.error("Failed to load CSV.")
        st.exception(e)
        st.stop()

    # Basic validations
    required_cols = {'Sequence', 'Taxonomy', 'Taxonomic_Level', 'Sample_ID', 'Read_Count'}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"CSV missing required columns. Found: {list(df.columns)}. Required: {list(required_cols)}")
        st.stop()

    # Preprocess: kmer
    status_msg.info("Converting sequences to k-mers…")
    df['kmer_sequence'] = df['Sequence'].apply(sequence_to_kmers)
    progress_placeholder.progress(12)

    if st.session_state.cancel_requested:
        st.warning("Run cancelled after k-mer conversion.")
        st.stop()

    # If label_map missing, fallback to CSV-inferred mapping but warn strongly
    if not label_to_species:
        known_species = df.loc[df['Taxonomy'].notna(), 'Taxonomy'].unique()
        label_to_species = {i: s for i, s in enumerate(known_species)}
        species_to_label = {s: i for i, s in label_to_species.items()}
        st.warning("No model label map found — using labels inferred from CSV. Add label_map.json for accurate mapping.")

    # Embeddings
    st.info("Computing embeddings (BERT)…")
    sequences_list = df['kmer_sequence'].tolist()
    embeddings = compute_embeddings(sequences_list, model_emb, tokenizer, device, batch_size=batch_size)
    if embeddings.shape[0] != len(df):
        st.warning("Embeddings length mismatch — filling with zeros for missing rows.")
        if embeddings.shape[0] < len(df):
            missing = len(df) - embeddings.shape[0]
            if embeddings.size == 0:
                hidden_size = getattr(getattr(model_emb, "config", None), "hidden_size", 768)
                embeddings = np.zeros((len(df), hidden_size))
            else:
                embeddings = np.vstack([embeddings, np.zeros((missing, embeddings.shape[1]))])
    df['embeddings'] = list(embeddings)
    progress_placeholder.progress(50)
    status_msg.success("Embeddings ready.")

    if st.session_state.cancel_requested:
        st.warning("Run cancelled after embeddings.")
        st.stop()

    # Predict missing taxonomy (with top-3)
    unassigned_mask = df['Taxonomy'].isna()
    if unassigned_mask.any():
        st.info(f"Predicting taxonomy for {int(unassigned_mask.sum())} unassigned sequences…")
        sequences_to_predict = df.loc[unassigned_mask, 'kmer_sequence'].tolist()
        logits = predict_logits(
            model_clf, tokenizer, device, sequences_to_predict,
            batch_size=batch_size, temperature=(temp_value if use_temp else 1.0)
        )
        probs = F.softmax(logits, dim=1).numpy()
        topk = np.argsort(-probs, axis=1)[:, :3]
        topk_probs = np.take_along_axis(probs, topk, axis=1)

        def idx_to_species(idx_row):
            return [label_to_species.get(int(i), f"Unknown_{int(i)}") for i in idx_row]

        topk_species = [idx_to_species(row) for row in topk]
        top1_probs = topk_probs[:, 0]

        predicted_species_top1 = []
        for species_name, p in zip([t[0] for t in topk_species], top1_probs):
            if float(p) < float(conf_threshold):
                predicted_species_top1.append(f"Uncertain_{species_name}")
            else:
                predicted_species_top1.append(species_name)

        # Write results
        df.loc[unassigned_mask, 'Taxonomy'] = predicted_species_top1
        df.loc[unassigned_mask, 'Taxonomic_Level'] = 'species_predicted'
        p1, p2, p3 = topk_probs[:, 0], topk_probs[:, 1], topk_probs[:, 2]
        pred1 = [t[0] for t in topk_species]
        pred2 = [t[1] for t in topk_species]
        pred3 = [t[2] for t in topk_species]

        df.loc[unassigned_mask, 'pred_1'] = pred1
        df.loc[unassigned_mask, 'pred_1_conf'] = p1
        df.loc[unassigned_mask, 'pred_2'] = pred2
        df.loc[unassigned_mask, 'pred_2_conf'] = p2
        df.loc[unassigned_mask, 'pred_3'] = pred3
        df.loc[unassigned_mask, 'pred_3_conf'] = p3

        status_msg.success("Prediction complete (top-3 confidences added).")
    else:
        status_msg.info("No missing taxonomy found in data.")
    progress_placeholder.progress(72)

    if st.session_state.cancel_requested:
        st.warning("Run cancelled after prediction.")
        st.stop()

    # Prepare final df (exclude references)
    final_df = df[df['Taxonomic_Level'] != 'reference'].copy()

    # HDBSCAN clustering for predicted species (novel detection)
    predicted_species_df = final_df[final_df['Taxonomic_Level'] == 'species_predicted'].copy()
    if not predicted_species_df.empty and len(predicted_species_df) > 10:
        st.info("Finding novel clusters with HDBSCAN…")
        pred_emb_array = np.vstack(predicted_species_df['embeddings'].values)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        predicted_species_df['novel_cluster'] = clusterer.fit_predict(pred_emb_array)
        final_df.loc[predicted_species_df.index, 'Taxonomy'] = (
            'Novel_Taxa_' + predicted_species_df['novel_cluster'].astype(str)
        )
        st.success("Novel cluster labels assigned.")
    else:
        st.info("Not enough predicted species to run novel-cluster detection.")
    progress_placeholder.progress(86)

    if st.session_state.cancel_requested:
        st.warning("Run cancelled after clustering.")
        st.stop()

    # Diversity metrics
    diversity_metrics = calculate_diversity_metrics(final_df)
    progress_placeholder.progress(95)

    # UMAP for novel taxa
    novel_taxa_only = final_df[final_df['Taxonomy'].str.startswith('Novel_Taxa_', na=False)].copy()
    if not novel_taxa_only.empty and len(novel_taxa_only) > 1:
        st.info("Computing UMAP for novel taxa visualization…")
        reducer = UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42, metric='cosine')
        novel_embeddings = np.vstack(novel_taxa_only['embeddings'].tolist())
        umap_result = reducer.fit_transform(novel_embeddings)
        final_df.loc[novel_taxa_only.index, 'umap_x'] = umap_result[:, 0]
        final_df.loc[novel_taxa_only.index, 'umap_y'] = umap_result[:, 1]
        st.success("UMAP ready.")
    progress_placeholder.progress(100)
    st.balloons()

    # =============================
    # Results UI (polished)
    # =============================
    st.success("Analysis complete — explore the insights below.")

    samples = final_df['Sample_ID'].unique().tolist()
    if not samples:
        st.error("No samples found after processing.")
        st.stop()
    selected_sample = st.selectbox("📍 Select a Sample Location", options=samples, index=0)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.markdown('<div class="kpi"><div class="label">Rows processed</div><div class="value">{:,}</div></div>'.format(len(final_df)), unsafe_allow_html=True)
    with colB:
        n_pred = int((final_df['Taxonomic_Level'] == 'species_predicted').sum())
        st.markdown(f'<div class="kpi"><div class="label">Predicted sequences</div><div class="value">{n_pred}</div></div>', unsafe_allow_html=True)
    with colC:
        n_novel = int(final_df['Taxonomy'].str.startswith('Novel_Taxa_', na=False).sum())
        st.markdown(f'<div class="kpi"><div class="label">Novel taxa points</div><div class="value">{n_novel}</div></div>', unsafe_allow_html=True)
    with colD:
        st.markdown(f'<div class="kpi"><div class="label">Batch size</div><div class="value">{batch_size}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌿 Ecosystem Metrics", "📈 Species Abundance", "🧭 Novel Taxa Discovery"])

    # TAB 1: Metrics
    with tab1:
        st.markdown('<div class="section-title">Ecosystem Health Metrics</div>', unsafe_allow_html=True)
        if selected_sample in diversity_metrics.index:
            metrics = diversity_metrics.loc[selected_sample]
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.metric("Shannon Index", f"{metrics['Shannon_Index']:.4f}")
            with c2:
                st.metric("Simpson Index", f"{metrics['Simpson_Index']:.4f}")
            with c3:
                st.caption("Shannon = richness+evenness; Simpson = dominance (lower is more even if using ∑p²).")

            sample_abund = final_df[final_df['Sample_ID'] == selected_sample] \
                                 .groupby('Taxonomy')['Read_Count'] \
                                 .sum().reset_index()
            st.markdown("**Species table for this sample**")
            st.dataframe(sample_abund.sort_values('Read_Count', ascending=False).reset_index(drop=True), height=280)
        else:
            st.info("Metrics not available for this sample.")

    # TAB 2: Species Abundance
    with tab2:
        st.markdown('<div class="section-title">Abundance Distribution</div>', unsafe_allow_html=True)
        sample_df = final_df[final_df['Sample_ID'] == selected_sample].copy()
        abundance_data = sample_df.groupby('Taxonomy')['Read_Count'] \
                                 .sum().reset_index() \
                                 .sort_values('Read_Count', ascending=False)

        max_k = max(5, len(abundance_data))
        top_k = st.slider("Show top K taxa", min_value=5, max_value=min(50, max_k), value=min(15, max_k))
        abundance_data_top = abundance_data.head(top_k)

        fig = plot_abundance_bar(abundance_data_top, selected_sample, top_k=top_k)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Search / download table**")
        q = st.text_input("Filter taxonomy (contains)", value="")
        filtered_table = abundance_data[abundance_data['Taxonomy'].str.contains(q, case=False, na=False)]
        st.dataframe(filtered_table, height=320)
        csv_bytes = filtered_table.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download filtered table (CSV)", csv_bytes, file_name=f"species_abundance_{selected_sample}.csv", use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("**Predicted sequences — top-3 confidences**")
        preds_sample = sample_df[sample_df['Taxonomic_Level'] == 'species_predicted'][[
            'Sequence', 'Taxonomy', 'pred_1','pred_1_conf','pred_2','pred_2_conf','pred_3','pred_3_conf','Read_Count'
        ]].copy()
        if not preds_sample.empty:
            preds_sample = preds_sample.sort_values('pred_1_conf', ascending=False)
            preds_sample['pred_1_conf'] = preds_sample['pred_1_conf'].apply(lambda x: f"{float(x):.3f}")
            preds_sample['pred_2_conf'] = preds_sample['pred_2_conf'].apply(lambda x: f"{float(x):.3f}")
            preds_sample['pred_3_conf'] = preds_sample['pred_3_conf'].apply(lambda x: f"{float(x):.3f}")
            st.dataframe(preds_sample.reset_index(drop=True), height=360)
        else:
            st.info("No predicted sequences in this sample.")

    # TAB 3: Novel taxa
    with tab3:
        st.markdown('<div class="section-title">Potential Novel Taxa (Unsupervised)</div>', unsafe_allow_html=True)
        novel_taxa_in_sample = final_df[
            (final_df['Taxonomy'].str.startswith('Novel_Taxa_', na=False)) &
            (final_df['Sample_ID'] == selected_sample)
        ].copy()

        if not novel_taxa_in_sample.empty and 'umap_x' in novel_taxa_in_sample.columns:
            fig2 = plot_novel_umap(novel_taxa_in_sample, selected_sample)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Novel taxa details**")
            cluster_table = novel_taxa_in_sample.groupby('Taxonomy').agg({
                'Read_Count': 'sum',
                'Sequence': 'count'
            }).rename(columns={'Sequence':'Count'}).reset_index().sort_values('Read_Count', ascending=False)
            st.dataframe(cluster_table, height=320)
        else:
            st.info("No novel taxa found in this sample for visualization.")

    # Footer raw data
    with st.expander("🗂️ Show full processed table"):
        st.dataframe(final_df.drop(columns=['embeddings'], errors='ignore').reset_index(drop=True), height=420)

    elapsed = time.time() - start_time
    st.success(f"Dashboard ready — re-run with a new file/model from the sidebar.  ⏱ Elapsed: {elapsed:.1f}s")

else:
    st.info("Use the **sidebar** → set data & model → click **Run analysis**.")
