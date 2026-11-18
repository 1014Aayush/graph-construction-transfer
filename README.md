# Will My GNN Survive a Rebuilt Graph?

Code for *Will My GNN Survive a Rebuilt Graph? Causes and Label-Free Checks for Graph-Construction Transfer* (A. Pokhrel, 2026).

## Files
| File | Purpose |
|---|---|
| `recipes.py` | Graph-construction recipes (kNN, PCA-kNN, PPR diffusion, threshold, rewiring) |
| `models.py` | GCN, GAT, GraphSAGE, GAT-sep, GCN-res, MLP |
| `diagnostics.py` | Label-free comparison measures and accuracy estimators |
| `controlled.py` | Controlled (one-factor) graph pairs |
| `run.py`, `run_v5.py`, `labelfree.py` | Experiment runners |
| `analyze_v4.py`, `analyze_v5.py`, `analyze_v7.py` | Analyses behind the paper's tables and figures |
| `notebooks/` | Kaggle notebooks used for the runs |
| `tests/` | Unit tests (`pytest tests`) |

## Usage
```bash
pip install -r requirements.txt
python run.py --help
```
Datasets are downloaded by PyTorch Geometric.
