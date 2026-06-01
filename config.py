"""
config.py — Merkezi Konfigürasyon
====================================
Cost-Based Optimization in Vector Databases (Senior Thesis)

Tüm yollar, hiper-parametreler ve özellik anahtarları burada
tanımlanır.  Projede kullanılan her modül bu dosyayı referans alır.
"""

from pathlib import Path

# ── Dataset Yolları ────────────────────────────────────────────────────────────
REVIEWS_PATH = Path("/Users/erengurkan/jobs/thesis/dataset/Electronics.json")
META_PATH    = Path("/Users/erengurkan/jobs/thesis/dataset/meta_Electronics.json")

# ── FAISS İndeks Dizini ────────────────────────────────────────────────────────
INDEX_DIR = Path("/Users/erengurkan/jobs/thesis/faiss_bench/index_data")

# ── Çıktı ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/erengurkan/jobs/thesis/bandit-cbo-unified/results")

# ── Ingestion ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 512            # Embedding batch boyutu
MAX_DOCUMENTS  = 400_000        # Incremental growth experiment (400k max)
TEXT_MAX_CHARS = 512            # Metin kırpma (embedding verimliliği)

# ── Embedder ───────────────────────────────────────────────────────────────────
# BERT-base model fine-tuned for sentence embeddings (768-dim)
EMBEDDING_MODEL  = "sentence-transformers/bert-base-nli-mean-tokens"
EMBEDDING_DIM    = 768          # BERT-base output dimension
HF_CACHE_DIR     = Path("/Users/erengurkan/jobs/thesis/bandit-cbo-unified/.hf-cache")

# ── FAISS HNSW Parametreleri ───────────────────────────────────────────────────
HNSW_M          = 32            # Her düğüme bağlanan komşu sayısı
HNSW_EF_CONSTR  = 200           # İndeks inşa sırasındaki ef değeri
HNSW_EF_SEARCH  = 128           # Arama sırasındaki ef değeri

# ── Benchmark ──────────────────────────────────────────────────────────────────
TOP_K_VALUES          = [10, 50]
SELECTIVITY_TARGETS   = [0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.21, 0.23, 0.25, 0.27, 0.30, 0.32, 0.34, 0.39, 0.61, 0.66, 0.68, 0.70, 0.73, 0.75, 0.77, 0.79, 0.82, 0.84, 0.86, 0.88, 0.91, 0.93, 0.95]
NUM_QUERY_SAMPLES     = 50
WARMUP_QUERIES        = 5
POST_FILTER_EXPANSION = 10      # Post-filter: top_k × expansion_factor

# ── Predictor Parametreleri ────────────────────────────────────────────────────
PREDICTOR_SAMPLE_RATIO = 0.03   # MiniBitmapPredictor örnekleme oranı

# ── Bitmap İndeks Alanları ─────────────────────────────────────────────────────
BITMAP_FIELDS = ["main_cat", "brand", "overall", "verified"]

# ── CBO (Contextual Bandit Optimizer) ──────────────────────────────────────────
CBO_R_TARGET       = 0.93      # Recall target for Soft Cliff SLA
CBO_RECALL_MARGIN  = 0.01      # Hard Cliff: Eğer recall < target - margin ise reward = 0
CBO_L_MAX          = 20.0      # Max acceptable latency (ms) — must be < brute force
CBO_BETA           = 10        # Recall penalty exponent (shock absorber)
CBO_ALPHA          = 0.1       # TD learning rate
CBO_SIGMA_LOWER    = 0.03      # Lower guardrail: force bitmap below this
CBO_SIGMA_UPPER    = 0.90      # Upper guardrail: force postfilter above this
CBO_CROSSOVER_HINT = 0.25      # Initial crossover estimate for bucket layout
CBO_N_EPOCHS       = 10        # Number of training epochs
CBO_EPSILON_MAX    = 0.3       # Max exploration rate (Exponential Decay)
CBO_DECAY_K        = 5.0       # Decay constant (Exponential Decay)
CBO_TAU_INIT       = 0.1       # Initial Softmax temperature
CBO_TOP_K          = 10        # Fixed top-K for CBO benchmark
CBO_FREEZE_MIN_VISITS = 15     # Minimum visits per bucket before trusting it to freeze
