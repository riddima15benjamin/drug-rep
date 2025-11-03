"""
GCN Training for Drug Repurposing Link Prediction
Trains a 2-layer GCN to predict chemical-disease links.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

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


class GCN(nn.Module):
    """2-layer Graph Convolutional Network."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, emb_dim: int = 128, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, emb_dim]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def load_data() -> Tuple[Data, Dict[str, int]]:
    """Load graph data and node mapping."""
    # Use weights_only=False since we trust our own generated files
    data = torch.load(OUTPUT_DIR / "graph_data.pt", weights_only=False)
    with open(OUTPUT_DIR / "node_to_idx.json", "r") as f:
        node_to_idx = json.load(f)
    
    logger.info(f"Loaded graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    return data, node_to_idx


def extract_chemical_disease_pairs(
    data: Data,
    node_to_idx: Dict[str, int]
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Extract all chemical-disease node pairs from the graph.
    
    Returns:
        Tuple of (positive_edges array, chemical_idx_map, disease_idx_map)
    """
    # Get chemical and disease node indices
    chemical_indices = []
    disease_indices = []
    
    for node_id, idx in node_to_idx.items():
        if node_id.startswith('chemical_'):
            chemical_indices.append(idx)
        elif node_id.startswith('disease_'):
            disease_indices.append(idx)
    
    chemical_indices = sorted(chemical_indices)
    disease_indices = sorted(disease_indices)
    
    logger.info(f"Found {len(chemical_indices)} chemicals and {len(disease_indices)} diseases")
    
    # Extract existing chemical-disease edges
    edge_index = data.edge_index.cpu().numpy()  # Move to CPU before numpy conversion
    positive_edges = []
    
    chemical_set = set(chemical_indices)
    disease_set = set(disease_indices)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        # Check if this is a chemical-disease edge
        if (src in chemical_set and dst in disease_set):
            positive_edges.append([src, dst])
        elif (src in disease_set and dst in chemical_set):
            positive_edges.append([dst, src])  # Reorder as chemical-disease
    
    # Remove duplicates
    positive_edges = np.unique(positive_edges, axis=0)
    
    logger.info(f"Found {len(positive_edges)} positive chemical-disease edges")
    
    # Create index mappings
    chem_idx_map = {idx: i for i, idx in enumerate(chemical_indices)}
    dis_idx_map = {idx: i for i, idx in enumerate(disease_indices)}
    
    return positive_edges, chem_idx_map, dis_idx_map


def split_edges(
    positive_edges: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split positive edges into train/val/test sets."""
    np.random.shuffle(positive_edges)
    
    n_edges = len(positive_edges)
    n_train = int(n_edges * train_ratio)
    n_val = int(n_edges * val_ratio)
    
    train_edges = positive_edges[:n_train]
    val_edges = positive_edges[n_train:n_train + n_val]
    test_edges = positive_edges[n_train + n_val:]
    
    logger.info(f"Split edges: {len(train_edges)} train, {len(val_edges)} val, {len(test_edges)} test")
    
    return train_edges, val_edges, test_edges


def sample_negative_edges(
    positive_edges: np.ndarray,
    chemical_indices: list,
    disease_indices: list,
    num_neg: int
) -> np.ndarray:
    """Sample negative edges that don't exist in positive set."""
    positive_set = set(map(tuple, positive_edges))
    negative_edges = []
    
    max_attempts = num_neg * 10
    attempts = 0
    
    while len(negative_edges) < num_neg and attempts < max_attempts:
        chem_idx = np.random.choice(chemical_indices)
        dis_idx = np.random.choice(disease_indices)
        
        if (chem_idx, dis_idx) not in positive_set:
            negative_edges.append([chem_idx, dis_idx])
            positive_set.add((chem_idx, dis_idx))  # Avoid duplicates
        
        attempts += 1
    
    return np.array(negative_edges)


def compute_link_scores(embeddings: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Compute link prediction scores using dot product."""
    src_emb = embeddings[edges[:, 0]]
    dst_emb = embeddings[edges[:, 1]]
    scores = (src_emb * dst_emb).sum(dim=1)
    return scores


def evaluate_model(
    model: GCN,
    data: Data,
    pos_edges: np.ndarray,
    neg_edges: np.ndarray,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on given edges."""
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index)
        
        # Compute scores
        pos_scores = compute_link_scores(embeddings, pos_edges).cpu().numpy()
        neg_scores = compute_link_scores(embeddings, neg_edges).cpu().numpy()
        
        # Combine labels and scores
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_score = np.concatenate([pos_scores, neg_scores])
        
        # Compute metrics
        auc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
    
    return auc, aupr


def train_epoch(
    model: GCN,
    data: Data,
    optimizer: torch.optim.Optimizer,
    train_pos_edges: np.ndarray,
    train_neg_edges: np.ndarray,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    embeddings = model(data.x, data.edge_index)
    
    # Compute scores
    pos_scores = compute_link_scores(embeddings, train_pos_edges)
    neg_scores = compute_link_scores(embeddings, train_neg_edges)
    
    # Labels
    pos_labels = torch.ones(len(pos_scores), device=device)
    neg_labels = torch.zeros(len(neg_scores), device=device)
    
    # Loss
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train GCN for drug repurposing")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting GCN Training")
    logger.info("="*60)
    logger.info(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, "
                f"hidden={args.hidden_dim}, emb={args.emb_dim}, dropout={args.dropout}")
    
    # Get device
    device = get_device()
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    data, node_to_idx = load_data()
    data = data.to(device)
    
    # Step 2: Extract chemical-disease pairs
    logger.info("\nStep 2: Extracting chemical-disease pairs...")
    positive_edges, chem_idx_map, dis_idx_map = extract_chemical_disease_pairs(data, node_to_idx)
    
    chemical_indices = list(chem_idx_map.keys())
    disease_indices = list(dis_idx_map.keys())
    
    # Step 3: Split edges
    logger.info("\nStep 3: Splitting edges...")
    train_pos, val_pos, test_pos = split_edges(positive_edges)
    
    # Sample negative edges
    train_neg = sample_negative_edges(train_pos, chemical_indices, disease_indices, len(train_pos))
    val_neg = sample_negative_edges(
        np.concatenate([train_pos, val_pos]), chemical_indices, disease_indices, len(val_pos)
    )
    test_neg = sample_negative_edges(
        positive_edges, chemical_indices, disease_indices, len(test_pos)
    )
    
    logger.info(f"Negative samples: {len(train_neg)} train, {len(val_neg)} val, {len(test_neg)} test")
    
    # Save test edges for evaluation
    np.save(OUTPUT_DIR / "test_pos_edges.npy", test_pos)
    np.save(OUTPUT_DIR / "test_neg_edges.npy", test_neg)
    
    # Step 4: Initialize model
    logger.info("\nStep 4: Initializing model...")
    input_dim = data.x.shape[1]
    model = GCN(input_dim, args.hidden_dim, args.emb_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 5: Training loop
    logger.info("\nStep 5: Training...")
    best_val_auc = 0
    patience_counter = 0
    train_log = []
    
    for epoch in tqdm(range(args.epochs), desc="Training"):
        # Train
        train_loss = train_epoch(model, data, optimizer, train_pos, train_neg, device)
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_auc, val_aupr = evaluate_model(model, data, val_pos, val_neg, device)
            
            # Log
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_auc': val_auc,
                'val_aupr': val_aupr
            }
            train_log.append(log_entry)
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, "
                       f"Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), OUTPUT_DIR / "model_best.pt")
                patience_counter = 0
                logger.info(f"  -> New best model saved (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save training log
    train_log_df = pd.DataFrame(train_log)
    train_log_df.to_csv(OUTPUT_DIR / "train_log.csv", index=False)
    
    # Step 6: Generate final embeddings
    logger.info("\nStep 6: Generating final embeddings...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "model_best.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()
    
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    logger.info(f"Saved embeddings: shape {embeddings.shape}")
    
    # Final test evaluation
    test_auc, test_aupr = evaluate_model(model, data, test_pos, test_neg, device)
    logger.info(f"\nFinal Test Performance:")
    logger.info(f"  Test AUC: {test_auc:.4f}")
    logger.info(f"  Test AUPR: {test_aupr:.4f}")
    
    # Save final metrics
    final_metrics = {
        'best_val_auc': best_val_auc,
        'test_auc': test_auc,
        'test_aupr': test_aupr
    }
    with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info("Output files:")
    logger.info("  - model_best.pt")
    logger.info("  - embeddings.npy")
    logger.info("  - train_log.csv")
    logger.info("  - final_metrics.json")


if __name__ == "__main__":
    main()

