# EUTERPE — Sparse Interaction Matrix Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Matrix Architecture:** Compressed Sparse Row (`scipy.sparse.csr_matrix`)  

---

## 1. Matrix Topology & Dimensionality

| Parameter | Value | Interpretation |
|---|:---:|---|
| **Users ($|\mathcal{U}|$)** | **884** | Eligible training listeners |
| **Items ($|\mathcal{I}|$)** | **443,375** | Eligible training track recordings |
| **Total Possible Matrix Cells** | **391,943,500** | $|\mathcal{U}| \times |\mathcal{I}|$ interaction space |
| **Confidence Weighting** | $c_{ui} = 1 + 40.0 \cdot \ln(1 + r_{ui})$ | Hu, Koren & Volinsky (2008) formulation |

---

## 2. Partition Sparsity & Density Analysis

| Matrix Split | Non-Zero Entries (`nnz`) | Matrix Density (%) | Matrix Sparsity (%) | In-Memory Footprint |
|---|:---:|:---:|:---:|:---:|
| **Train Matrix (`train_matrix.npz`)** | **2,776,274** | **0.7083%** | **99.2917%** | **21.18 MB** |
| **Validation (`validation_matrix.npz`)** | **592,152** | **0.1511%** | **99.8489%** | ~4.52 MB |
| **Test Matrix (`test_matrix.npz`)** | **507,516** | **0.1295%** | **99.8705%** | ~3.87 MB |

---

## 3. Engineering Assessment & Index Mapping Integrity
- All user and track indices strictly adhere to 0-indexed contiguous integers.
- Exact bidirectional lookups (`user_to_idx.json`, `item_to_idx.json`, `item_metadata.json`) are preserved in `artifacts/mappings/` for sub-millisecond API retrieval and model serving.
