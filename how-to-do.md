## How to Run the Notebook

Follow these steps to execute the project successfully in **Jupyter Notebook** or **JupyterLab**.

---

### Step 1 – Clone the Repository

---
title: Setup Guide
description: Repository setup

```bash
git clone https://github.com/2024ad05357/advanced-rag-enterprise.git
cd advanced-rag-enterprise
```

---
 

### Step 2 – Create and Activate a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 – Install Required Libraries

Run the following command once:

```bash
pip install datasets sentence-transformers faiss-cpu rank-bm25 pandas pyarrow scikit-learn matplotlib seaborn jupyter
```

---

### Step 4 – Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook:

```
Advanced_Enterprise_RAG_Assignment.ipynb
```

---

### Step 5 – Execute the Notebook Sequentially

Run the notebook **from top to bottom** using **Kernel → Restart & Run All**.

The notebook is designed to cache expensive operations locally, so subsequent runs are significantly faster.

---

### First-Time Execution

During the first run, the notebook will automatically:

1. Download the **MedQuAD dataset** from Hugging Face.
2. Save it locally as `data/medquad.parquet`.
3. Download the embedding and reranker models.
4. Cache the models inside the `models/` directory.
5. Create the processed corpus and semantic chunks.

This may take **2–5 minutes** depending on the internet connection and CPU performance.

---

### Subsequent Runs

On later executions, the notebook will **reuse cached artifacts** and will not download the dataset or models again. You should see messages such as:

```
Loading cached MedQuAD dataset...
Loading cached embedding model...
Loading cached reranker model...
```

---

### Expected Directory Structure After Successful Execution

```
advanced-rag-enterprise/
├── Advanced_Enterprise_RAG_Assignment.ipynb
├── data/
│   ├── medquad.parquet
│   ├── processed/
│   │   └── medquad_processed.parquet
│   └── chunks/
│       └── medquad_chunks.parquet
├── models/
│   ├── all-MiniLM-L6-v2/
│   └── ms-marco-MiniLM-L-6-v2/
└── README.md
```

---

### Recommended Notebook Execution Order

| Section | Purpose                                                |
| ------- | ------------------------------------------------------ |
| 1–3     | Assignment description and dataset justification       |
| 4       | Environment setup                                      |
| 5       | Dataset download and caching                           |
| 6       | Model loading and caching                              |
| 7       | Corpus preparation and metadata construction           |
| 8       | Chunking strategy comparison                           |
| 9–12    | Retrieval, reranking, agentic workflow, and evaluation |
| 13–15   | Discussion, conclusion, and references                 |

---

### Quick Restart Point

If the notebook has already been executed up to **Task 1**, you can restart the kernel and continue from:

```
Section 9 – Task 2: Hybrid Retrieval
```

The previously generated semantic chunk corpus stored in `data/chunks/medquad_chunks.parquet` will be reused automatically.

---

### Troubleshooting

#### `ModuleNotFoundError: sentence_transformers`

```bash
pip install sentence-transformers
```

#### `ModuleNotFoundError: faiss`

```bash
pip install faiss-cpu
```

#### Slow First Run

* Ensure internet access is available for the initial model download.
* The embedding model is approximately **90 MB** and is downloaded only once.

---

### Recommended System Configuration

| Component  | Recommended                |
| ---------- | -------------------------- |
| Python     | 3.10+                      |
| RAM        | 8 GB or higher             |
| CPU        | Modern dual-core or better |
| GPU        | Not required               |
| Disk Space | ~1 GB free                 |

The notebook has been intentionally designed to run **entirely on CPU without requiring Google Colab or GPU acceleration**, making it suitable for university virtual lab environments and standard development laptops.
