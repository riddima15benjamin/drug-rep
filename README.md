# Drug Repurposing for Accelerated Therapeutic Discovery

A complete end-to-end machine learning system for drug repurposing using **Graph Convolutional Networks (GCN)** on the **Comparative Toxicogenomics Database (CTD)**.

## 🎯 Project Overview

This project implements a graph-based deep learning approach to predict novel drug-disease associations by learning from chemical-gene-disease interactions. The system:

- Processes heterogeneous biomedical interaction data from CTD
- Constructs a knowledge graph with chemicals, genes, and diseases
- Trains a 2-layer GCN for link prediction
- Generates and interprets drug repurposing predictions
- Provides an interactive Streamlit demo

## 📊 Dataset

The project uses the **CTD (Comparative Toxicogenomics Database)** available at https://ctdbase.org/

Expected CSV files in `./dataset/`:

- Chemical-Disease interactions
- Chemical-Gene interactions
- Gene-Disease interactions

The system auto-detects CSV formats and adapts to different column naming conventions.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# Install other requirements
pip install -r requirements.txt
```

**Note:** For GPU support, adjust PyTorch installation based on your CUDA version. See [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) installation guides.

### 3. Download CTD Data

1. Visit https://ctdbase.org/downloads/
2. Download the following files:
   - `CTD_chemicals_diseases.csv`
   - `CTD_chem_gene_ixns.csv` (optional)
   - `CTD_genes_diseases.csv` (optional)
3. Place them in `./dataset/` directory

```bash
mkdir -p dataset
# Copy your downloaded CTD CSV files here
```

## 📝 Usage

### Step-by-Step Pipeline

#### 1. Data Preprocessing

```bash
python preprocess.py --top_chemicals 150 --top_diseases 100 --top_genes 200
```

**Options:**

- `--top_chemicals`: Number of most frequent chemicals to select (default: 150)
- `--top_diseases`: Number of most frequent diseases to select (default: 100)
- `--top_genes`: Number of most frequent genes to select (default: 200)

**Outputs:**

- `./outputs/chemical_disease.csv`
- `./outputs/chemical_gene.csv`
- `./outputs/gene_disease.csv`
- `./outputs/node_features.npy`
- `./outputs/node_to_idx.json`
- `./outputs/top_nodes.json`

#### 2. Graph Construction

```bash
python build_graph.py
```

**Outputs:**

- `./outputs/graph_data.pt` (PyTorch Geometric Data object)
- `./outputs/graph.gpickle` (NetworkX graph for analysis)

#### 3. Model Training

```bash
python train_gcn.py --epochs 100 --lr 0.001
```

**Options:**

- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--emb_dim`: Embedding dimension (default: 128)
- `--dropout`: Dropout rate (default: 0.5)
- `--patience`: Early stopping patience (default: 10)

**Outputs:**

- `./outputs/model_best.pt` (trained model)
- `./outputs/embeddings.npy` (node embeddings)
- `./outputs/train_log.csv` (training history)
- `./outputs/final_metrics.json`

#### 4. Evaluation

```bash
python evaluate.py
```

**Outputs:**

- `./outputs/test_metrics.csv` / `test_metrics.json`
- `./outputs/predictions.csv` (top-100 predictions per disease)
- `./outputs/interpretation_top10.csv` (mechanistic interpretations)
- `./outputs/plots/roc_curve.png`
- `./outputs/plots/pr_curve.png`

#### 5. Interactive Demo

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to interact with the system.

### One-Click Execution (Notebook)

For a complete end-to-end run, use the Jupyter notebook:

```bash
jupyter notebook notebook.ipynb
```

The notebook reproduces all steps: preprocessing → graph building → training → evaluation in a single flow.

## 📁 Project Structure

```
drug-repurposing/
├── dataset/                      # CTD CSV files (user-provided)
│   ├── CTD_chemicals_diseases.csv
│   └── ...
├── outputs/                      # Generated outputs
│   ├── model_best.pt
│   ├── embeddings.npy
│   ├── predictions.csv
│   ├── plots/
│   └── ...
├── results/
│   └── report-slides/           # Reports and slides
│       ├── report.pdf
│       └── demo-slides.pptx
├── tests/
│   └── test_smoke.sh            # Smoke tests
├── preprocess.py                # Data preprocessing
├── build_graph.py               # Graph construction
├── train_gcn.py                 # GCN training
├── evaluate.py                  # Model evaluation
├── app.py                       # Streamlit demo
├── notebook.ipynb               # End-to-end notebook
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Model Architecture

### Graph Convolutional Network (GCN)

```
Input Features (node features)
    ↓
GCNConv(in_dim → 128)
    ↓
ReLU + Dropout(0.5)
    ↓
GCNConv(128 → 128)
    ↓
Node Embeddings
```

### Link Prediction

- **Task:** Predict chemical-disease associations
- **Method:** Dot-product scoring between node embeddings
- **Loss:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Negative Sampling:** 1:1 ratio with positive edges

### Training Details

- **Optimizer:** Adam
- **Learning Rate:** 1e-3
- **Weight Decay:** 1e-5
- **Early Stopping:** Based on validation AUC (patience=10)
- **Data Split:** 70% train / 15% val / 15% test

## 📈 Results

Sample performance metrics (actual results depend on your data):

| Metric       | Value |
| ------------ | ----- |
| Test AUC     | 0.85+ |
| Test AUPR    | 0.78+ |
| Precision@10 | 0.65+ |

## 🔍 Interpretation

The system provides mechanistic interpretations by:

1. Finding shortest paths between predicted drug-disease pairs
2. Identifying intermediate gene nodes
3. Visualizing interaction subgraphs

## 🛠️ Troubleshooting

### PyTorch Geometric Installation

If you encounter issues installing PyTorch Geometric:

```bash
# For CPU
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# For CUDA 11.8
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Out of Memory

If you run out of memory during training:

1. Reduce the number of selected nodes in preprocessing:

   ```bash
   python preprocess.py --top_chemicals 100 --top_diseases 50 --top_genes 100
   ```

2. Reduce batch size or use neighbor sampling (for large graphs)

### Missing CSV Files

If CSV detection fails:

- Check column names in your CSV files
- Refer to `./outputs/schema_assumptions.txt` to see what was inferred
- Manually rename columns to match: `ChemicalID`, `DiseaseID`, `GeneID`

## 📚 References

1. **CTD Database:** Davis AP, et al. "The Comparative Toxicogenomics Database." Nucleic Acids Res. 2023.
2. **Graph Convolutional Networks:** Kipf TN, Welling M. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
3. **Drug Repurposing:** Pushpakom S, et al. "Drug repurposing: progress, challenges and recommendations." Nat Rev Drug Discov. 2019.

## 📄 License

This project is for educational and research purposes. Please cite CTD and relevant papers if you use this system in your research.

## 👥 Contact

For questions or issues, please open a GitHub issue or contact the development team.

---

**Note:** This is a research prototype. Predictions should be validated through biological experiments before clinical application.
