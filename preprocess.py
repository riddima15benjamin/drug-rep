"""
Data Preprocessing for Drug Repurposing Project
Auto-detects CTD CSV files and creates filtered subgraph with node features.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Setup directories
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("./dataset")

# Setup logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "log.txt", encoding='utf-8'),
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


def log_metadata():
    """Log Python version and timestamp."""
    metadata = {
        "python_version": sys.version,
        "timestamp": datetime.now().isoformat(),
        "script": "preprocess.py"
    }
    with open(OUTPUT_DIR / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    logger.info("Metadata logged")


def detect_csv_files() -> Dict[str, Path]:
    """
    Auto-detect CTD CSV files in ./dataset/ directory.
    
    Returns:
        Dictionary mapping interaction type to file path
    """
    csv_files = {}
    schema_info = []
    
    if not DATA_DIR.exists():
        logger.error(f"Dataset directory {DATA_DIR} does not exist!")
        return csv_files
    
    for csv_file in DATA_DIR.glob("*.csv"):
        logger.info(f"Found CSV file: {csv_file.name}")
        
        # Read first few rows to detect schema
        try:
            # Try reading with header first
            df_sample = pd.read_csv(csv_file, nrows=5, comment='#')
            columns = list(df_sample.columns)
            
            # Check if file has no header (numeric column names indicate no header)
            has_header = not all(str(col).isdigit() or str(col).startswith('Unnamed') for col in columns)
            
            if not has_header:
                # CTD files often have no header, infer from filename
                logger.info(f"  No header detected, inferring from filename")
                df_sample = pd.read_csv(csv_file, nrows=5, comment='#', header=None)
                columns = [f"Column_{i}" for i in range(len(df_sample.columns))]
            
            schema_info.append(f"\n{csv_file.name}:\n  Columns: {columns}\n  Has Header: {has_header}\n  Sample:\n{df_sample.to_string()}\n")
            
            # Detect file type based on column names
            columns_lower = [col.lower() for col in columns]
            
            if any('chemical' in col and 'disease' in col for col in columns_lower):
                csv_files['chemical_disease'] = csv_file
                logger.info(f"  -> Detected as chemical-disease interactions")
            elif any('chemical' in col and 'gene' in col for col in columns_lower):
                csv_files['chemical_gene'] = csv_file
                logger.info(f"  -> Detected as chemical-gene interactions")
            elif any('gene' in col and ('disease' in col or 'phenotype' in col) for col in columns_lower):
                csv_files['gene_disease'] = csv_file
                logger.info(f"  -> Detected as gene-disease interactions")
            else:
                # Try to infer from column names or filename
                filename_lower = csv_file.name.lower()
                
                # Check filename for hints
                if 'chemical' in filename_lower and 'disease' in filename_lower:
                    csv_files['chemical_disease'] = csv_file
                    logger.info(f"  -> Inferred as chemical-disease from filename")
                elif 'chem' in filename_lower and 'gene' in filename_lower:
                    csv_files['chemical_gene'] = csv_file
                    logger.info(f"  -> Inferred as chemical-gene from filename")
                elif 'gene' in filename_lower and 'disease' in filename_lower:
                    csv_files['gene_disease'] = csv_file
                    logger.info(f"  -> Inferred as gene-disease from filename")
                # Try to infer from first two column names
                elif len(columns) >= 2:
                    col0, col1 = columns[0].lower(), columns[1].lower()
                    if 'chemical' in col0 or 'drug' in col0:
                        if 'disease' in col1:
                            csv_files['chemical_disease'] = csv_file
                            logger.info(f"  -> Inferred as chemical-disease")
                        elif 'gene' in col1:
                            csv_files['chemical_gene'] = csv_file
                            logger.info(f"  -> Inferred as chemical-gene")
                    elif 'gene' in col0 and 'disease' in col1:
                        csv_files['gene_disease'] = csv_file
                        logger.info(f"  -> Inferred as gene-disease")
                    else:
                        logger.warning(f"  -> Could not determine file type")
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")
    
    # Save schema information
    with open(OUTPUT_DIR / "schema_sample.txt", "w") as f:
        f.write("=== Detected CSV Files ===\n")
        f.write(f"Total files found: {len(csv_files)}\n")
        for interaction_type, path in csv_files.items():
            f.write(f"- {interaction_type}: {path.name}\n")
        f.write("\n=== Schema Samples ===\n")
        f.write("".join(schema_info))
    
    return csv_files


def load_interactions(csv_files: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load interaction data from detected CSV files.
    
    Returns:
        Tuple of (chemical_disease_df, chemical_gene_df, gene_disease_df)
    """
    dfs = {}
    assumptions = []
    
    for interaction_type, csv_path in csv_files.items():
        try:
            # Check if file has header
            df_check = pd.read_csv(csv_path, nrows=1, comment='#')
            has_header = not all(str(col).isdigit() or str(col).startswith('Unnamed') for col in df_check.columns)
            
            if has_header:
                # Read CSV with header
                df = pd.read_csv(csv_path, comment='#')
                df.columns = [col.strip() for col in df.columns]
                logger.info(f"Loaded {interaction_type}: {len(df)} rows (with header)")
            else:
                # Read CSV without header - CTD standard format
                df = pd.read_csv(csv_path, comment='#', header=None)
                logger.info(f"Loaded {interaction_type}: {len(df)} rows (no header, using CTD format)")
            
            # Detect ID columns
            if interaction_type == 'chemical_disease':
                if has_header:
                    # Find chemical and disease ID columns
                    chem_col = None
                    disease_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'chemical' in col_lower and 'id' in col_lower:
                            chem_col = col
                        elif 'disease' in col_lower and 'id' in col_lower:
                            disease_col = col
                    
                    # Fallback: use first two columns
                    if chem_col is None:
                        chem_col = df.columns[0]
                        assumptions.append(f"chemical_disease: Using column '{chem_col}' as ChemicalID")
                    if disease_col is None:
                        disease_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        assumptions.append(f"chemical_disease: Using column '{disease_col}' as DiseaseID")
                else:
                    # CTD standard format: col 1=ChemicalID, col 4=DiseaseID
                    chem_col = 1
                    disease_col = 4
                    assumptions.append(f"chemical_disease: Using CTD format (col 1=ChemicalID, col 4=DiseaseID)")
                
                df = df[[chem_col, disease_col]].rename(columns={
                    chem_col: 'ChemicalID',
                    disease_col: 'DiseaseID'
                })
                df = df.dropna()
                df['ChemicalID'] = df['ChemicalID'].astype(str)
                df['DiseaseID'] = df['DiseaseID'].astype(str)
                
            elif interaction_type == 'chemical_gene':
                if has_header:
                    chem_col = None
                    gene_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'chemical' in col_lower and 'id' in col_lower:
                            chem_col = col
                        elif 'gene' in col_lower and ('id' in col_lower or 'symbol' in col_lower):
                            gene_col = col
                    
                    if chem_col is None:
                        chem_col = df.columns[0]
                        assumptions.append(f"chemical_gene: Using column '{chem_col}' as ChemicalID")
                    if gene_col is None:
                        gene_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        assumptions.append(f"chemical_gene: Using column '{gene_col}' as GeneID")
                else:
                    # CTD standard format: col 1=ChemicalID, col 4=GeneSymbol
                    chem_col = 1
                    gene_col = 4
                    assumptions.append(f"chemical_gene: Using CTD format (col 1=ChemicalID, col 4=GeneSymbol)")
                
                df = df[[chem_col, gene_col]].rename(columns={
                    chem_col: 'ChemicalID',
                    gene_col: 'GeneID'
                })
                df = df.dropna()
                df['ChemicalID'] = df['ChemicalID'].astype(str)
                df['GeneID'] = df['GeneID'].astype(str)
                
            elif interaction_type == 'gene_disease':
                if has_header:
                    gene_col = None
                    disease_col = None
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'gene' in col_lower and ('id' in col_lower or 'symbol' in col_lower):
                            gene_col = col
                        elif ('disease' in col_lower or 'phenotype' in col_lower) and 'id' in col_lower:
                            disease_col = col
                    
                    if gene_col is None:
                        gene_col = df.columns[0]
                        assumptions.append(f"gene_disease: Using column '{gene_col}' as GeneID")
                    if disease_col is None:
                        disease_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                        assumptions.append(f"gene_disease: Using column '{disease_col}' as DiseaseID")
                else:
                    # CTD standard format: col 0=GeneSymbol, col 2=DiseaseID
                    gene_col = 0
                    disease_col = 2
                    assumptions.append(f"gene_disease: Using CTD format (col 0=GeneSymbol, col 2=DiseaseID)")
                
                df = df[[gene_col, disease_col]].rename(columns={
                    gene_col: 'GeneID',
                    disease_col: 'DiseaseID'
                })
                df = df.dropna()
                df['GeneID'] = df['GeneID'].astype(str)
                df['DiseaseID'] = df['DiseaseID'].astype(str)
            
            dfs[interaction_type] = df
            logger.info(f"Processed {interaction_type}: {len(df)} valid interactions")
            
        except Exception as e:
            logger.error(f"Error processing {interaction_type}: {e}")
            dfs[interaction_type] = pd.DataFrame()
    
    # Save assumptions
    if assumptions:
        with open(OUTPUT_DIR / "schema_assumptions.txt", "w") as f:
            f.write("=== Schema Assumptions ===\n")
            for assumption in assumptions:
                f.write(f"- {assumption}\n")
    
    return (
        dfs.get('chemical_disease', pd.DataFrame()),
        dfs.get('chemical_gene', pd.DataFrame()),
        dfs.get('gene_disease', pd.DataFrame())
    )


def select_top_nodes(
    chem_dis_df: pd.DataFrame,
    chem_gene_df: pd.DataFrame,
    gene_dis_df: pd.DataFrame,
    top_k_chemicals: int,
    top_k_diseases: int,
    top_k_genes: int
) -> Dict[str, List[str]]:
    """
    Select most frequent nodes based on edge counts.
    
    Returns:
        Dictionary with 'chemicals', 'diseases', 'genes' lists
    """
    # Count occurrences
    chemical_counter = Counter()
    disease_counter = Counter()
    gene_counter = Counter()
    
    # Chemical-Disease
    if not chem_dis_df.empty:
        chemical_counter.update(chem_dis_df['ChemicalID'])
        disease_counter.update(chem_dis_df['DiseaseID'])
    
    # Chemical-Gene
    if not chem_gene_df.empty:
        chemical_counter.update(chem_gene_df['ChemicalID'])
        gene_counter.update(chem_gene_df['GeneID'])
    
    # Gene-Disease
    if not gene_dis_df.empty:
        gene_counter.update(gene_dis_df['GeneID'])
        disease_counter.update(gene_dis_df['DiseaseID'])
    
    # Calculate total interactions
    total_interactions = len(chem_dis_df) + len(chem_gene_df) + len(gene_dis_df)
    logger.info(f"Total interactions before filtering: {total_interactions}")
    
    # Adjust selection size if dataset is small
    if total_interactions < 50000:
        logger.info("Dataset is small (<50k interactions), increasing selection sizes")
        top_k_chemicals = max(top_k_chemicals, 300)
        top_k_diseases = max(top_k_diseases, 200)
        top_k_genes = max(top_k_genes, 400)
    
    # Select top nodes
    top_chemicals = [chem for chem, _ in chemical_counter.most_common(top_k_chemicals)]
    top_diseases = [dis for dis, _ in disease_counter.most_common(top_k_diseases)]
    top_genes = [gene for gene, _ in gene_counter.most_common(top_k_genes)]
    
    logger.info(f"Selected nodes: {len(top_chemicals)} chemicals, {len(top_diseases)} diseases, {len(top_genes)} genes")
    
    selected_nodes = {
        'chemicals': top_chemicals,
        'diseases': top_diseases,
        'genes': top_genes
    }
    
    # Save to JSON
    with open(OUTPUT_DIR / "top_nodes.json", "w") as f:
        json.dump(selected_nodes, f, indent=2)
    
    return selected_nodes


def filter_interactions(
    chem_dis_df: pd.DataFrame,
    chem_gene_df: pd.DataFrame,
    gene_dis_df: pd.DataFrame,
    selected_nodes: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter interactions to only include selected nodes."""
    chemicals_set = set(selected_nodes['chemicals'])
    diseases_set = set(selected_nodes['diseases'])
    genes_set = set(selected_nodes['genes'])
    
    # Filter chemical-disease
    if not chem_dis_df.empty:
        chem_dis_filtered = chem_dis_df[
            chem_dis_df['ChemicalID'].isin(chemicals_set) &
            chem_dis_df['DiseaseID'].isin(diseases_set)
        ]
    else:
        chem_dis_filtered = pd.DataFrame(columns=['ChemicalID', 'DiseaseID'])
    
    # Filter chemical-gene
    if not chem_gene_df.empty:
        chem_gene_filtered = chem_gene_df[
            chem_gene_df['ChemicalID'].isin(chemicals_set) &
            chem_gene_df['GeneID'].isin(genes_set)
        ]
    else:
        chem_gene_filtered = pd.DataFrame(columns=['ChemicalID', 'GeneID'])
    
    # Filter gene-disease
    if not gene_dis_df.empty:
        gene_dis_filtered = gene_dis_df[
            gene_dis_df['GeneID'].isin(genes_set) &
            gene_dis_df['DiseaseID'].isin(diseases_set)
        ]
    else:
        gene_dis_filtered = pd.DataFrame(columns=['GeneID', 'DiseaseID'])
    
    logger.info(f"Filtered interactions: {len(chem_dis_filtered)} chem-dis, "
                f"{len(chem_gene_filtered)} chem-gene, {len(gene_dis_filtered)} gene-dis")
    
    # Save filtered CSVs
    chem_dis_filtered.to_csv(OUTPUT_DIR / "chemical_disease.csv", index=False)
    chem_gene_filtered.to_csv(OUTPUT_DIR / "chemical_gene.csv", index=False)
    gene_dis_filtered.to_csv(OUTPUT_DIR / "gene_disease.csv", index=False)
    
    return chem_dis_filtered, chem_gene_filtered, gene_dis_filtered


def compute_node_features(
    selected_nodes: Dict[str, List[str]],
    chem_dis_df: pd.DataFrame,
    chem_gene_df: pd.DataFrame,
    gene_dis_df: pd.DataFrame
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute node features for all selected nodes.
    Features: node_type one-hot + normalized degree
    
    Returns:
        Tuple of (feature_matrix, node_to_idx mapping)
    """
    # Create node to index mapping
    all_nodes = []
    node_types = []  # 0: chemical, 1: gene, 2: disease
    
    for chem in selected_nodes['chemicals']:
        all_nodes.append(('chemical', chem))
        node_types.append(0)
    
    for gene in selected_nodes['genes']:
        all_nodes.append(('gene', gene))
        node_types.append(1)
    
    for disease in selected_nodes['diseases']:
        all_nodes.append(('disease', disease))
        node_types.append(2)
    
    node_to_idx = {f"{ntype}_{nid}": idx for idx, (ntype, nid) in enumerate(all_nodes)}
    
    # Compute degrees
    degree_dict = {}
    for ntype, nid in all_nodes:
        degree = 0
        if ntype == 'chemical':
            degree += sum(chem_dis_df['ChemicalID'] == nid)
            degree += sum(chem_gene_df['ChemicalID'] == nid)
        elif ntype == 'gene':
            degree += sum(chem_gene_df['GeneID'] == nid)
            degree += sum(gene_dis_df['GeneID'] == nid)
        elif ntype == 'disease':
            degree += sum(chem_dis_df['DiseaseID'] == nid)
            degree += sum(gene_dis_df['DiseaseID'] == nid)
        degree_dict[f"{ntype}_{nid}"] = degree
    
    # Create feature matrix
    num_nodes = len(all_nodes)
    features = []
    
    for idx, (ntype, nid) in enumerate(all_nodes):
        # One-hot encoding for node type (3 dimensions)
        node_type_onehot = [0, 0, 0]
        node_type_onehot[node_types[idx]] = 1
        
        # Degree
        degree = degree_dict[f"{ntype}_{nid}"]
        
        # Combine features
        feat = node_type_onehot + [degree]
        features.append(feat)
    
    features = np.array(features, dtype=np.float32)
    
    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    logger.info(f"Created feature matrix: shape {features.shape}")
    
    # Save
    np.save(OUTPUT_DIR / "node_features.npy", features)
    with open(OUTPUT_DIR / "node_to_idx.json", "w") as f:
        json.dump(node_to_idx, f, indent=2)
    
    return features, node_to_idx


def create_manifest():
    """Create a manifest of all created files."""
    files_created = []
    for f in OUTPUT_DIR.glob("*"):
        if f.is_file():
            files_created.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "type": f.suffix
            })
    
    manifest_df = pd.DataFrame(files_created)
    manifest_df.to_csv(OUTPUT_DIR / "manifest.csv", index=False)
    logger.info(f"Created manifest with {len(files_created)} files")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CTD data for drug repurposing")
    parser.add_argument("--top_chemicals", type=int, default=150, help="Number of top chemicals to select")
    parser.add_argument("--top_diseases", type=int, default=100, help="Number of top diseases to select")
    parser.add_argument("--top_genes", type=int, default=200, help="Number of top genes to select")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting Data Preprocessing")
    logger.info("="*60)
    
    # Log metadata
    log_metadata()
    
    # Step 1: Detect CSV files
    logger.info("\nStep 1: Detecting CSV files...")
    csv_files = detect_csv_files()
    if not csv_files:
        logger.error("No CSV files detected! Exiting.")
        sys.exit(1)
    
    # Step 2: Load interactions
    logger.info("\nStep 2: Loading interactions...")
    chem_dis_df, chem_gene_df, gene_dis_df = load_interactions(csv_files)
    
    # Step 3: Select top nodes
    logger.info("\nStep 3: Selecting top nodes by frequency...")
    selected_nodes = select_top_nodes(
        chem_dis_df, chem_gene_df, gene_dis_df,
        args.top_chemicals, args.top_diseases, args.top_genes
    )
    
    # Step 4: Filter interactions
    logger.info("\nStep 4: Filtering interactions...")
    chem_dis_filtered, chem_gene_filtered, gene_dis_filtered = filter_interactions(
        chem_dis_df, chem_gene_df, gene_dis_df, selected_nodes
    )
    
    # Step 5: Compute node features
    logger.info("\nStep 5: Computing node features...")
    features, node_to_idx = compute_node_features(
        selected_nodes, chem_dis_filtered, chem_gene_filtered, gene_dis_filtered
    )
    
    # Step 6: Create manifest
    logger.info("\nStep 6: Creating manifest...")
    create_manifest()
    
    logger.info("\n" + "="*60)
    logger.info("Preprocessing Complete!")
    logger.info("="*60)
    logger.info(f"Total nodes: {len(node_to_idx)}")
    logger.info(f"Total edges: {len(chem_dis_filtered) + len(chem_gene_filtered) + len(gene_dis_filtered)}")
    logger.info(f"Feature dimension: {features.shape[1]}")
    logger.info("\nOutput files saved in ./outputs/")


if __name__ == "__main__":
    main()

