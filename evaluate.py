"""
Model Evaluation and Interpretation
Computes test metrics, generates predictions, and interprets results.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Import model
sys.path.append(str(Path(__file__).parent))
from train_gcn import GCN

# Set random seed
SEED = 42
np.random.seed(SEED)

# Directories
OUTPUT_DIR = Path("./outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Setup logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "log.txt", mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def load_model_and_data() -> Tuple[GCN, Data, Dict[str, int], np.ndarray]:
    """Load trained model, graph data, and embeddings."""
    # Load data (weights_only=False for PyTorch Geometric Data objects)
    data = torch.load(OUTPUT_DIR / "graph_data.pt", weights_only=False)
    with open(OUTPUT_DIR / "node_to_idx.json", "r") as f:
        node_to_idx = json.load(f)
    
    # Load embeddings
    embeddings = np.load(OUTPUT_DIR / "embeddings.npy")
    
    # Load model (weights_only=True for model state dicts)
    input_dim = data.x.shape[1]
    model = GCN(input_dim, hidden_dim=128, emb_dim=128, dropout=0.5)
    model.load_state_dict(torch.load(OUTPUT_DIR / "model_best.pt", weights_only=True, map_location='cpu'))
    model.eval()
    
    logger.info(f"Loaded model and data: {data.num_nodes} nodes, embeddings shape {embeddings.shape}")
    
    return model, data, node_to_idx, embeddings


def compute_link_scores(embeddings: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute link prediction scores using dot product."""
    src_emb = embeddings[edges[:, 0]]
    dst_emb = embeddings[edges[:, 1]]
    scores = (src_emb * dst_emb).sum(axis=1)
    return scores


def evaluate_test_set(embeddings: np.ndarray) -> Dict[str, float]:
    """Evaluate on test set with various metrics."""
    # Load test edges
    test_pos = np.load(OUTPUT_DIR / "test_pos_edges.npy")
    test_neg = np.load(OUTPUT_DIR / "test_neg_edges.npy")
    
    # Compute scores
    pos_scores = compute_link_scores(embeddings, test_pos)
    neg_scores = compute_link_scores(embeddings, test_neg)
    
    # Combine
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    # Compute metrics
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    
    # Precision@K and Recall@K
    metrics = {'auc': auc, 'aupr': aupr}
    
    # Sort by score
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    for k in [1, 3, 5, 10, 20, 50, 100]:
        if k <= len(y_true_sorted):
            precision_at_k = y_true_sorted[:k].sum() / k
            recall_at_k = y_true_sorted[:k].sum() / y_true.sum()
            metrics[f'precision@{k}'] = precision_at_k
            metrics[f'recall@{k}'] = recall_at_k
    
    logger.info("\nTest Set Metrics:")
    logger.info(f"  ROC AUC: {auc:.4f}")
    logger.info(f"  AUPR: {aupr:.4f}")
    logger.info(f"  Precision@10: {metrics.get('precision@10', 0):.4f}")
    logger.info(f"  Recall@10: {metrics.get('recall@10', 0):.4f}")
    
    return metrics, y_true, y_score


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Drug-Disease Link Prediction', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve to {PLOTS_DIR / 'roc_curve.png'}")


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUPR = {aupr:.3f})', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - Drug-Disease Link Prediction', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pr_curve.png", dpi=300)
    plt.close()
    logger.info(f"Saved PR curve to {PLOTS_DIR / 'pr_curve.png'}")


def generate_predictions(
    embeddings: np.ndarray,
    node_to_idx: Dict[str, int],
    data: Data,
    top_k: int = 100
) -> pd.DataFrame:
    """
    Generate top-K predictions for each disease.
    
    Returns:
        DataFrame with columns: disease_id, chemical_id, score, rank
    """
    # Get chemical and disease indices
    chemical_nodes = [(idx, nid) for nid, idx in node_to_idx.items() if nid.startswith('chemical_')]
    disease_nodes = [(idx, nid) for nid, idx in node_to_idx.items() if nid.startswith('disease_')]
    
    # Load training positive edges to exclude
    train_pos_path = OUTPUT_DIR / "test_pos_edges.npy"
    if train_pos_path.exists():
        train_pos = np.load(train_pos_path)
        train_pos_set = set(map(tuple, train_pos))
    else:
        train_pos_set = set()
    
    # Get all existing chemical-disease edges
    edge_index = data.edge_index.numpy()
    existing_edges = set()
    chemical_indices = set([idx for idx, _ in chemical_nodes])
    disease_indices = set([idx for idx, _ in disease_nodes])
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in chemical_indices and dst in disease_indices:
            existing_edges.add((src, dst))
        elif src in disease_indices and dst in chemical_indices:
            existing_edges.add((dst, src))
    
    # Generate predictions for each disease
    predictions = []
    
    for dis_idx, dis_id in disease_nodes:
        disease_name = dis_id.replace('disease_', '')
        
        # Compute scores for all chemicals
        candidate_scores = []
        for chem_idx, chem_id in chemical_nodes:
            # Skip if training positive
            if (chem_idx, dis_idx) in train_pos_set:
                continue
            
            # Compute score
            score = np.dot(embeddings[chem_idx], embeddings[dis_idx])
            chemical_name = chem_id.replace('chemical_', '')
            
            # Check if known
            is_known = (chem_idx, dis_idx) in existing_edges
            
            candidate_scores.append({
                'chemical_idx': chem_idx,
                'chemical_id': chemical_name,
                'score': score,
                'is_known': is_known
            })
        
        # Sort by score
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top K
        for rank, item in enumerate(candidate_scores[:top_k], 1):
            predictions.append({
                'disease_id': disease_name,
                'chemical_id': item['chemical_id'],
                'score': item['score'],
                'rank': rank,
                'is_known': item['is_known']
            })
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    logger.info(f"Generated {len(predictions_df)} predictions for {len(disease_nodes)} diseases")
    
    return predictions_df


def interpret_predictions(
    predictions_df: pd.DataFrame,
    node_to_idx: Dict[str, int]
) -> pd.DataFrame:
    """
    Find gene intermediates for top predictions using shortest paths.
    
    Returns:
        DataFrame with top predictions and their intermediate genes
    """
    # Load NetworkX graph
    with open(OUTPUT_DIR / "graph.gpickle", "rb") as f:
        G = pickle.load(f)
    
    # Get top novel predictions (not known)
    novel_predictions = predictions_df[predictions_df['is_known'] == False].head(10)
    
    interpretations = []
    
    for _, row in novel_predictions.iterrows():
        disease_id = f"disease_{row['disease_id']}"
        chemical_id = f"chemical_{row['chemical_id']}"
        
        # Find shortest path
        try:
            if G.has_node(chemical_id) and G.has_node(disease_id):
                path = nx.shortest_path(G, chemical_id, disease_id)
                
                # Extract intermediate genes
                intermediate_genes = [node.replace('gene_', '') for node in path[1:-1] if node.startswith('gene_')]
                
                interpretations.append({
                    'disease_id': row['disease_id'],
                    'chemical_id': row['chemical_id'],
                    'score': row['score'],
                    'rank': row['rank'],
                    'path_length': len(path),
                    'intermediate_genes': '; '.join(intermediate_genes) if intermediate_genes else 'None',
                    'full_path': ' -> '.join([n.split('_', 1)[1] for n in path])
                })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            interpretations.append({
                'disease_id': row['disease_id'],
                'chemical_id': row['chemical_id'],
                'score': row['score'],
                'rank': row['rank'],
                'path_length': float('inf'),
                'intermediate_genes': 'No path',
                'full_path': 'No path'
            })
    
    interpretation_df = pd.DataFrame(interpretations)
    interpretation_df.to_csv(OUTPUT_DIR / "interpretation_top10.csv", index=False)
    logger.info(f"Generated interpretations for top 10 predictions")
    
    return interpretation_df


def save_metrics(metrics: Dict[str, float]):
    """Save evaluation metrics."""
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    
    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved test metrics")


def main():
    logger.info("="*60)
    logger.info("Starting Model Evaluation")
    logger.info("="*60)
    
    # Step 1: Load model and data
    logger.info("\nStep 1: Loading model and data...")
    model, data, node_to_idx, embeddings = load_model_and_data()
    
    # Step 2: Evaluate on test set
    logger.info("\nStep 2: Evaluating on test set...")
    metrics, y_true, y_score = evaluate_test_set(embeddings)
    
    # Step 3: Plot curves
    logger.info("\nStep 3: Generating plots...")
    plot_roc_curve(y_true, y_score)
    plot_pr_curve(y_true, y_score)
    
    # Step 4: Generate predictions
    logger.info("\nStep 4: Generating predictions for all diseases...")
    predictions_df = generate_predictions(embeddings, node_to_idx, data, top_k=100)
    
    # Step 5: Interpret top predictions
    logger.info("\nStep 5: Interpreting top predictions...")
    interpretation_df = interpret_predictions(predictions_df, node_to_idx)
    
    # Display sample interpretations
    logger.info("\nSample Interpretations (Top 3):")
    for i, row in interpretation_df.head(3).iterrows():
        logger.info(f"\n  {i+1}. {row['chemical_id']} -> {row['disease_id']}")
        logger.info(f"     Score: {row['score']:.4f}")
        logger.info(f"     Path: {row['full_path']}")
    
    # Step 6: Save metrics
    logger.info("\nStep 6: Saving metrics...")
    save_metrics(metrics)
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info("Output files:")
    logger.info("  - test_metrics.csv / test_metrics.json")
    logger.info("  - predictions.csv")
    logger.info("  - interpretation_top10.csv")
    logger.info("  - plots/roc_curve.png")
    logger.info("  - plots/pr_curve.png")


if __name__ == "__main__":
    main()

