"""
Streamlit Demo for Drug Repurposing System
Interactive visualization of drug-disease predictions.
"""

import sys
import json
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Setup paths
OUTPUT_DIR = Path("./outputs")

# Page config
st.set_page_config(
    page_title="Drug Repurposing for Accelerated Therapeutic Discovery",
    page_icon="💊",
    layout="wide"
)


@st.cache_data
def load_predictions():
    """Load precomputed predictions."""
    try:
        predictions_df = pd.read_csv(OUTPUT_DIR / "predictions.csv")
        return predictions_df
    except FileNotFoundError:
        st.error(f"Predictions file not found at {OUTPUT_DIR / 'predictions.csv'}. Please run evaluate.py first.")
        return None


@st.cache_data
def load_interpretations():
    """Load interpretation data."""
    try:
        interp_df = pd.read_csv(OUTPUT_DIR / "interpretation_top10.csv")
        return interp_df
    except FileNotFoundError:
        return None


@st.cache_data
def load_graph():
    """Load NetworkX graph."""
    try:
        with open(OUTPUT_DIR / "graph.gpickle", "rb") as f:
            G = pickle.load(f)
        return G
    except FileNotFoundError:
        st.error(f"Graph file not found at {OUTPUT_DIR / 'graph.gpickle'}. Please run build_graph.py first.")
        return None


@st.cache_data
def load_embeddings():
    """Load node embeddings."""
    try:
        embeddings = np.load(OUTPUT_DIR / "embeddings.npy")
        with open(OUTPUT_DIR / "node_to_idx.json", "r") as f:
            node_to_idx = json.load(f)
        return embeddings, node_to_idx
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_metrics():
    """Load test metrics."""
    try:
        with open(OUTPUT_DIR / "test_metrics.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        return None


def load_disease_mapping():
    """Load disease ID to name mapping."""
    try:
        with open(OUTPUT_DIR / "prediction_disease_id_to_name.json", "r") as f:
            mapping = json.load(f)
        return mapping
    except FileNotFoundError:
        return {}


def get_disease_list(predictions_df, disease_mapping):
    """Get unique disease names from predictions with mapping."""
    if predictions_df is not None:
        disease_ids = sorted(predictions_df['disease_id'].unique())
        # Create list of tuples (display_name, disease_id) for selectbox
        disease_options = []
        for disease_id in disease_ids:
            disease_name = disease_mapping.get(disease_id, disease_id)
            disease_options.append((f"{disease_name} ({disease_id})", disease_id))
        return disease_options
    return []


def filter_predictions(predictions_df, disease_id, top_k):
    """Filter predictions for selected disease."""
    disease_preds = predictions_df[predictions_df['disease_id'] == disease_id].head(top_k)
    return disease_preds


def create_subgraph_visualization(G, disease_id, top_chemicals, max_genes=5):
    """Create interactive subgraph with disease, chemicals, and connecting genes."""
    if G is None:
        return None
    
    disease_node = f"disease_{disease_id}"
    
    # Create subgraph
    nodes_to_include = [disease_node]
    edges_to_include = []
    
    # Add chemicals and their paths
    for chem_id in top_chemicals:
        chem_node = f"chemical_{chem_id}"
        if chem_node in G and disease_node in G:
            nodes_to_include.append(chem_node)
            
            # Try to find shortest path
            try:
                path = nx.shortest_path(G, chem_node, disease_node)
                # Add intermediate genes (limited)
                gene_nodes = [n for n in path[1:-1] if n.startswith('gene_')][:max_genes]
                nodes_to_include.extend(gene_nodes)
                
                # Add edges from path
                for i in range(len(path) - 1):
                    if path[i] in nodes_to_include and path[i+1] in nodes_to_include:
                        edges_to_include.append((path[i], path[i+1]))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
    
    # Create subgraph
    subgraph = G.subgraph(nodes_to_include)
    
    # Layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Create plotly figure
    edge_trace = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Node traces
    node_traces = {}
    for node_type, color, name in [
        ('disease', '#FF6B6B', 'Disease'),
        ('chemical', '#4ECDC4', 'Chemical'),
        ('gene', '#95E1D3', 'Gene')
    ]:
        x, y, text = [], [], []
        for node in subgraph.nodes():
            if node.startswith(node_type):
                node_x, node_y = pos[node]
                x.append(node_x)
                y.append(node_y)
                text.append(node.split('_', 1)[1])
        
        if x:
            node_traces[node_type] = go.Scatter(
                x=x, y=y,
                mode='markers+text',
                text=text,
                textposition='top center',
                marker=dict(size=15, color=color, line=dict(width=2, color='white')),
                name=name,
                hoverinfo='text'
            )
    
    # Create figure
    fig = go.Figure(data=edge_trace + list(node_traces.values()))
    
    fig.update_layout(
        title=f"Subgraph for Disease: {disease_id}",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_embedding_visualization(embeddings, node_to_idx, highlighted_chemicals, method='PCA'):
    """Create 2D projection of chemical embeddings."""
    if embeddings is None or node_to_idx is None:
        return None
    
    # Extract chemical embeddings
    chemical_indices = []
    chemical_labels = []
    
    for node_id, idx in node_to_idx.items():
        if node_id.startswith('chemical_'):
            chemical_indices.append(idx)
            chemical_labels.append(node_id.replace('chemical_', ''))
    
    chemical_embeddings = embeddings[chemical_indices]
    
    # Dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(chemical_embeddings) - 1))
    
    coords = reducer.fit_transform(chemical_embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'chemical': chemical_labels,
        'highlighted': [c in highlighted_chemicals for c in chemical_labels]
    })
    
    # Create plotly figure
    fig = go.Figure()
    
    # Background chemicals
    bg_df = df[~df['highlighted']]
    if not bg_df.empty:
        fig.add_trace(go.Scatter(
            x=bg_df['x'], y=bg_df['y'],
            mode='markers',
            marker=dict(size=5, color='lightgray', opacity=0.5),
            name='Other Chemicals',
            hovertext=bg_df['chemical'],
            hoverinfo='text'
        ))
    
    # Highlighted chemicals
    hl_df = df[df['highlighted']]
    if not hl_df.empty:
        fig.add_trace(go.Scatter(
            x=hl_df['x'], y=hl_df['y'],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='star'),
            text=hl_df['chemical'],
            textposition='top center',
            name='Predicted Chemicals',
            hovertext=hl_df['chemical'],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title=f"Chemical Embeddings ({method})",
        xaxis_title=f"{method} Component 1",
        yaxis_title=f"{method} Component 2",
        height=500,
        showlegend=True
    )
    
    return fig


def main():
    # Title
    st.title(" Drug Repurposing for Accelerated Therapeutic Discovery")
    st.markdown("### Graph Convolutional Network for Drug-Disease Link Prediction")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        predictions_df = load_predictions()
        interp_df = load_interpretations()
        G = load_graph()
        embeddings, node_to_idx = load_embeddings()
        metrics = load_metrics()
        disease_mapping = load_disease_mapping()
    
    if predictions_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Disease selection
    disease_options = get_disease_list(predictions_df, disease_mapping)
    if disease_options:
        selected_option = st.sidebar.selectbox(
            "Select Disease",
            disease_options,
            format_func=lambda x: x[0],  # Display the name part
            help="Choose a disease to view predictions"
        )
        # Extract just the disease ID for processing
        selected_disease = selected_option[1]
    else:
        selected_disease = None
    
    # Top-K slider
    top_k = st.sidebar.slider(
        "Number of Predictions",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of top predictions to display"
    )
    
    # Embedding method
    emb_method = st.sidebar.radio(
        "Embedding Visualization",
        ["PCA", "t-SNE"],
        help="Dimensionality reduction method"
    )
    
    st.sidebar.markdown("---")
    
    # Display metrics
    if metrics:
        st.sidebar.subheader("📊 Model Performance")
        st.sidebar.metric("Test AUC", f"{metrics.get('auc', 0):.3f}")
        st.sidebar.metric("Test AUPR", f"{metrics.get('aupr', 0):.3f}")
        if 'precision@10' in metrics:
            st.sidebar.metric("Precision@10", f"{metrics['precision@10']:.3f}")
    
    # Main content
    if selected_disease:
        disease_name = disease_mapping.get(selected_disease, selected_disease)
        st.subheader(f"🎯 Predictions for Disease: {disease_name}")
        st.caption(f"Disease ID: `{selected_disease}`")
        
        # Filter predictions
        disease_preds = filter_predictions(predictions_df, selected_disease, top_k)
        
        if disease_preds.empty:
            st.warning("No predictions available for this disease.")
            return
        
        # Display table
        st.markdown("#### Top Predicted Chemicals")
        
        # Format table
        display_df = disease_preds.copy()
        display_df['score'] = display_df['score'].round(4)
        display_df['is_known'] = display_df['is_known'].map({True: '✓ Known', False: '✗ Novel'})
        display_df = display_df.rename(columns={
            'chemical_id': 'Chemical ID',
            'score': 'Score',
            'rank': 'Rank',
            'is_known': 'Status'
        })
        
        # Add interpretation if available
        if interp_df is not None:
            interp_dict = {}
            for _, row in interp_df.iterrows():
                if row['disease_id'] == selected_disease:
                    interp_dict[row['chemical_id']] = row['intermediate_genes']
            
            display_df['Intermediate Genes'] = display_df['Chemical ID'].map(interp_dict).fillna('N/A')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Download button
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name=f"predictions_{selected_disease}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌐 Interaction Subgraph")
            with st.spinner("Generating graph..."):
                top_chemicals = disease_preds['chemical_id'].head(5).tolist()
                fig_graph = create_subgraph_visualization(G, selected_disease, top_chemicals)
                if fig_graph:
                    st.plotly_chart(fig_graph, use_container_width=True)
                else:
                    st.info("Graph visualization not available")
        
        with col2:
            st.markdown("#### 📍 Embedding Space")
            with st.spinner("Computing embeddings..."):
                highlighted = disease_preds['chemical_id'].head(10).tolist()
                fig_emb = create_embedding_visualization(embeddings, node_to_idx, highlighted, emb_method)
                if fig_emb:
                    st.plotly_chart(fig_emb, use_container_width=True)
                else:
                    st.info("Embedding visualization not available")
        
        # Additional insights
        st.markdown("---")
        st.subheader("💡 Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_novel = (disease_preds['is_known'] == '✗ Novel').sum()
            st.metric("Novel Predictions", n_novel)
        
        with col2:
            avg_score = disease_preds['score'].mean()
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col3:
            n_total = len(disease_preds)
            st.metric("Total Predictions", n_total)


if __name__ == "__main__":
    main()

