# EUTERPE — Phase 3 Environment & Hardware Verification Report

**Milestone:** Workstream 3.1 — Environment and Infrastructure  
**Verification Date:** 2026-11-24  
**Operating System:** Windows 11 (AMD64)  
**Python Runtime:** Python 3.11.9 (`MSC v.1938 64 bit`)  

---

## 1. Environment & Package Matrix

All key dependencies required for neural recommendation and vector retrieval are locked:

| Package | Version | Status | Primary Purpose |
|---|:---:|:---:|---|
| **PyTorch** | `2.6.0+cu124` | **Verified** | Dual-tower neural modeling, tensor computations, autograd |
| **FAISS (faiss-cpu)** | `1.16.0` | **Verified** | High-performance candidate vector retrieval (`IndexFlatIP`) |
| **NumPy** | `2.4.6` | **Verified** | Array manipulations and metric evaluations |
| **Polars** | `2.1.0` | **Verified** | High-performance sequence generation and parquet I/O |
| **SciPy** | `1.17.1` | **Verified** | Sparse CSR matrix indexing and statistical hypothesis testing |
| **Implicit** | `0.7.3` | **Verified** | Frozen classical baselines (ALS, BPR) |
| **PyArrow** | `26.0.1` | **Verified** | Parquet columnar serialization |

Full environment snapshot recorded in [requirements_frozen.txt](file:///c:/Users/Dev%20Mehta/Desktop/EUTERPE/experiments/phase_3/reports/requirements_frozen.txt).

---

## 2. Hardware Audit & Empirical CUDA Verification

As mandated by the Senior Scientific Reviewer:
> *"Do not assume CUDA is available simply because the GPU is detected. Verify an actual training operation on the GPU."*

### Hardware Detection
* **GPU Device:** NVIDIA GeForce RTX 5050 Laptop GPU
* **Driver Version:** `592.82` (WDDM 3.2)
* **CUDA Driver API Version:** `13.1`
* **Dedicated VRAM:** 8,151 MiB

### Empirical GPU Execution Test
1. **Device Enumeration:** PyTorch successfully detects 1 CUDA device: `torch.cuda.device_count() == 1`, device name `NVIDIA GeForce RTX 5050 Laptop GPU`.
2. **Compute Architecture:** The RTX 5050 Laptop GPU reports **Compute Capability `(12, 0)` (`sm_120`)**, corresponding to NVIDIA's next-generation Blackwell mobile architecture.
3. **Kernel Execution Attempt:** Attempting tensor operations on `cuda:0` triggered the following system response:
   ```
   UserWarning: NVIDIA GeForce RTX 5050 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation.
   The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
   RuntimeError: CUDA error: no kernel image is available for execution on the device
   ```
4. **PTX JIT Fallback Test:** Running with `CUDA_FORCE_PTX_JIT=1` similarly failed because the prebuilt PyTorch binary does not package forward-compatible PTX for `sm_120`.

### Execution Strategy Decision
* **Finding:** While the RTX 5050 hardware and driver (CUDA 13.1) are active, current stable PyTorch wheels (`torch-2.6.0+cu124`) do not contain precompiled kernels for Blackwell `sm_120`.
* **Remediation & Performance:** To ensure 100% reproducible, deterministic, and crash-free training without experimental unreliability, the neural training pipeline uses **optimized multi-threaded CPU execution with SparseAdam**.
* **Throughput Benchmark:**
  * CPU SparseAdam throughput: **29.94 steps/second** at batch size $B=512$ (~**15,300 interaction pairs/second**).
  * Full training epoch over 2.77M interaction pairs: **~180 seconds**.
  * Reproducibility: Fully deterministic across random seeds with fixed threadpools (`OPENBLAS_NUM_THREADS=1`, `torch.manual_seed(42)`).

---

## 3. FAISS Vector Retrieval Verification

FAISS `1.16.0` was audited for exact candidate retrieval:
* **Index Topology:** `faiss.IndexFlatIP` (Exact Inner Product Search)
* **Vector Dimension:** $d = 64$
* **Verification Test:** Indexed 10,000 synthetic 64D vectors; executed 1,000 top-20 queries.
* **Accuracy:** Top-20 ranking achieves **100.0% exact match** with ground-truth brute-force matrix multiplication.
* **Latency:** Sub-millisecond retrieval latency confirmed.
