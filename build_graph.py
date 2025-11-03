"""
Graph Construction for Drug Repurposing Project
Converts heterogeneous CTD interactions into a homogeneous graph for PyTorch Geometric.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Directories
OUTPUT_DIR = Path("./outputs")

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


def load_preprocessed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, Dict[str, int]]:
    """Load preprocessed data files."""
    logger.info("Loading preprocessed data...")
    
    # Load filtered interactions
    chem_dis_df = pd.read_csv(OUTPUT_DIR / "chemical_disease.csv")
    chem_gene_df = pd.read_csv(OUTPUT_DIR / "chemical_gene.csv")
    gene_dis_df = pd.read_csv(OUTPUT_DIR / "gene_disease.csv")
    
    logger.info(f"Loaded interactions: {len(chem_dis_df)} chem-dis, "
                f"{len(chem_gene_df)} chem-gene, {len(gene_dis_df)} gene-dis")
    
    # Load node features and mapping
    node_features = np.load(OUTPUT_DIR / "node_features.npy")
    with open(OUTPUT_DIR / "node_to_idx.json", "r") as f:
        node_to_idx = json.load(f)
    
    logger.info(f"Loaded {len(node_to_idx)} nodes with {node_features.shape[1]} features")
    
    return chem_dis_df, chem_gene_df, gene_dis_df, node_features, node_to_idx


def build_edge_index(
    chem_dis_df: pd.DataFrame,
    chem_gene_df: pd.DataFrame,
    gene_dis_df: pd.DataFrame,
    node_to_idx: Dict[str, int]
) -> Tuple[torch.Tensor, list]:
    """
    Build edge index tensor for PyTorch Geometric.
    Creates undirected edges (adds both directions).
    
    Returns:
        Tuple of (edge_index tensor, list of edge types for reference)
    """
    edges = []
    edge_types = []  # For reference: 0=chem-dis, 1=chem-gene, 2=gene-dis
    
    # Chemical-Disease edges
    for _, row in chem_dis_df.iterrows():
        chem_id = f"chemical_{row['ChemicalID']}"
        dis_id = f"disease_{row['DiseaseID']}"
        
        if chem_id in node_to_idx and dis_id in node_to_idx:
            idx1 = node_to_idx[chem_id]
            idx2 = node_to_idx[dis_id]
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])  # Undirected
            edge_types.extend([0, 0])
    
    # Chemical-Gene edges
    for _, row in chem_gene_df.iterrows():
        chem_id = f"chemical_{row['ChemicalID']}"
        gene_id = f"gene_{row['GeneID']}"
        
        if chem_id in node_to_idx and gene_id in node_to_idx:
            idx1 = node_to_idx[chem_id]
            idx2 = node_to_idx[gene_id]
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])  # Undirected
            edge_types.extend([1, 1])
    
    # Gene-Disease edges
    for _, row in gene_dis_df.iterrows():
        gene_id = f"gene_{row['GeneID']}"
        dis_id = f"disease_{row['DiseaseID']}"
        
        if gene_id in node_to_idx and dis_id in node_to_idx:
            idx1 = node_to_idx[gene_id]
            idx2 = node_to_idx[dis_id]
            edges.append([idx1, idx2])
            edges.append([idx2, idx1])  # Undirected
            edge_types.extend([2, 2])
    
    if not edges:
        logger.error("No valid edges found!")
        return torch.zeros((2, 0), dtype=torch.long), []
    
    # Convert to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    logger.info(f"Built edge index: {edge_index.shape[1]} edges (undirected)")
    
    return edge_index, edge_types


def create_node_type_tensor(node_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Create tensor of node types.
    0 = chemical, 1 = gene, 2 = disease
    """
    num_nodes = len(node_to_idx)
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    
    for node_id, idx in node_to_idx.items():
        if node_id.startswith('chemical_'):
            node_types[idx] = 0
        elif node_id.startswith('gene_'):
            node_types[idx] = 1
        elif node_id.startswith('disease_'):
            node_types[idx] = 2
    
    return node_types


def create_pytorch_geometric_data(
    node_features: np.ndarray,
    edge_index: torch.Tensor,
    node_types: torch.Tensor
) -> Data:
    """Create PyTorch Geometric Data object."""
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        node_type=node_types,
        num_nodes=len(node_features)
    )
    
    logger.info(f"Created PyG Data object:")
    logger.info(f"  Nodes: {data.num_nodes}")
    logger.info(f"  Edges: {data.edge_index.shape[1]}")
    logger.info(f"  Features: {data.x.shape}")
    
    return data


def create_networkx_graph(
    node_to_idx: Dict[str, int],
    chem_dis_df: pd.DataFrame,
    chem_gene_df: pd.DataFrame,
    gene_dis_df: pd.DataFrame
) -> nx.Graph:
    """Create NetworkX graph for analysis and visualization."""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node_id, idx in node_to_idx.items():
        if node_id.startswith('chemical_'):
            node_type = 'chemical'
            label = node_id.replace('chemical_', '')
        elif node_id.startswith('gene_'):
            node_type = 'gene'
            label = node_id.replace('gene_', '')
        else:
            node_type = 'disease'
            label = node_id.replace('disease_', '')
        
        G.add_node(node_id, node_type=node_type, label=label, idx=idx)
    
    # Add edges
    for _, row in chem_dis_df.iterrows():
        chem_id = f"chemical_{row['ChemicalID']}"
        dis_id = f"disease_{row['DiseaseID']}"
        if chem_id in node_to_idx and dis_id in node_to_idx:
            G.add_edge(chem_id, dis_id, edge_type='chemical_disease')
    
    for _, row in chem_gene_df.iterrows():
        chem_id = f"chemical_{row['ChemicalID']}"
        gene_id = f"gene_{row['GeneID']}"
        if chem_id in node_to_idx and gene_id in node_to_idx:
            G.add_edge(chem_id, gene_id, edge_type='chemical_gene')
    
    for _, row in gene_dis_df.iterrows():
        gene_id = f"gene_{row['GeneID']}"
        dis_id = f"disease_{row['DiseaseID']}"
        if gene_id in node_to_idx and dis_id in node_to_idx:
            G.add_edge(gene_id, dis_id, edge_type='gene_disease')
    
    logger.info(f"Created NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def save_graph_data(data: Data, graph: nx.Graph):
    """Save graph data to files."""
    # Save PyTorch Geometric data
    torch.save(data, OUTPUT_DIR / "graph_data.pt")
    logger.info(f"Saved PyTorch Geometric data to {OUTPUT_DIR / 'graph_data.pt'}")
    logger.info(f"Note: Use weights_only=False when loading this file due to PyG Data objects")
    
    # Save NetworkX graph
    with open(OUTPUT_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved NetworkX graph to {OUTPUT_DIR / 'graph.gpickle'}")


def main():
    logger.info("="*60)
    logger.info("Starting Graph Construction")
    logger.info("="*60)
    
    # Step 1: Load preprocessed data
    logger.info("\nStep 1: Loading preprocessed data...")
    chem_dis_df, chem_gene_df, gene_dis_df, node_features, node_to_idx = load_preprocessed_data()
    
    # Step 2: Build edge index
    logger.info("\nStep 2: Building edge index...")
    edge_index, edge_types = build_edge_index(chem_dis_df, chem_gene_df, gene_dis_df, node_to_idx)
    
    # Step 3: Create node type tensor
    logger.info("\nStep 3: Creating node type tensor...")
    node_types = create_node_type_tensor(node_to_idx)
    
    # Step 4: Create PyTorch Geometric Data
    logger.info("\nStep 4: Creating PyTorch Geometric Data object...")
    data = create_pytorch_geometric_data(node_features, edge_index, node_types)
    
    # Step 5: Create NetworkX graph
    logger.info("\nStep 5: Creating NetworkX graph...")
    graph = create_networkx_graph(node_to_idx, chem_dis_df, chem_gene_df, gene_dis_df)
    
    # Step 6: Save
    logger.info("\nStep 6: Saving graph data...")
    save_graph_data(data, graph)
    
    logger.info("\n" + "="*60)
    logger.info("Graph Construction Complete!")
    logger.info("="*60)
    logger.info("Output files:")
    logger.info("  - graph_data.pt (PyTorch Geometric)")
    logger.info("  - graph.gpickle (NetworkX)")


if __name__ == "__main__":
    main()

