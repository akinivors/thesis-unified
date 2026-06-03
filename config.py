"""
config.py — Merkezi Konfigürasyon
====================================
Cost-Based Optimization in Vector Databases (Senior Thesis)

Tüm yollar, hiper-parametreler ve özellik anahtarları burada
tanımlanır.  Projede kullanılan her modül bu dosyayı referans alır.
"""

from pathlib import Path

# ── Proje Kökü (bu dosyanın bulunduğu dizin) ───────────────────────────────────
PROJECT_ROOT = Path(__file__).parent

# ── Dataset Yolları ────────────────────────────────────────────────────────────
# Raw dataset files (only needed for ingest/build_index.py)
REVIEWS_PATH = PROJECT_ROOT / "dataset" / "Electronics.json"
META_PATH    = PROJECT_ROOT / "dataset" / "meta_Electronics.json"

# ── FAISS İndeks Dizini ────────────────────────────────────────────────────────
# Directory containing: hnsw.index, embeddings.npy, id_mappings.pkl, all_metadatas.pkl
# Run build_index.py once to populate this from your existing bandit-cbo data.
INDEX_DIR = PROJECT_ROOT / "index_data"

# ── Çıktı ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results"

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
HNSW_EF_SEARCH  = 128           # Baseline ef for unfiltered search (PostFilter)
# Adaptive ef for IDSelector pre-filter: min(max(EF_QUALITY, 3*k/sel), EF_MAX)
HNSW_EF_QUALITY = 512           # Minimum ef floor for filtered (IDSelector) searches
HNSW_EF_MAX     = 8192          # Hard cap on ef to prevent runaway latency

# ── Benchmark ──────────────────────────────────────────────────────────────────
TOP_K_VALUES          = [10, 50]
SELECTIVITY_TARGETS   = [0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.21, 0.23, 0.25, 0.27, 0.30, 0.32, 0.34, 0.39, 0.61, 0.66, 0.68, 0.70, 0.73, 0.75, 0.77, 0.79, 0.82, 0.84, 0.86, 0.88, 0.91, 0.93, 0.95]
NUM_QUERY_SAMPLES     = 50
WARMUP_QUERIES        = 5
POST_FILTER_EXPANSION = 100     # Post-filter base expansion: top_k × expansion_factor
                                # Adaptive formula also applies: max(100, int(2/selectivity))

# ── Predictor Parametreleri ────────────────────────────────────────────────────
PREDICTOR_SAMPLE_RATIO = 0.03   # MiniBitmapPredictor örnekleme oranı

# ── Bitmap İndeks Alanları ─────────────────────────────────────────────────────
BITMAP_FIELDS = ["main_cat", "brand", "overall", "verified"]

# ── CBO (Contextual Bandit Optimizer) ──────────────────────────────────────────
CBO_R_TARGET       = 0.93      # Recall target for Soft Cliff SLA
CBO_RECALL_MARGIN  = 0.01      # Hard Cliff: Eğer recall < target - margin ise reward = 0
CBO_L_MAX          = 100.0     # Max acceptable latency (ms). Overridden at runtime
                               # by compute_l_max() when n_docs is known; this constant
                               # is the calibrated baseline for a 200k-document corpus.
CBO_L_MAX_BASE     = 100.0     # Baseline L_MAX (ms) at CBO_L_MAX_REF_DOCS corpus size
CBO_L_MAX_REF_DOCS = 200_000   # Reference corpus size for L_MAX calibration
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
CBO_FREEZE_MAX_GAP  = 7       # Max unvisited buckets bridged in non-adjacent freeze check
                               # 7 covers the 26%-30% filter gap in the 671k corpus
N_WARMUP           = 5000     # Queries before Phase-2 bucket resampling is triggered.
                               # 2000 was insufficient for the reward margin at the 671k
                               # crossover (~0.035 at 30% selectivity) to converge before
                               # Phase 2 fires.  5000 gives ~160 visits/bucket in Phase 1,
                               # enough to resolve margins as narrow as 0.03 reliably.
N_WARMUP_BASE      = 5000     # Immutable baseline for compute_n_warmup.  Never mutate
                               # this constant — it is the fixed reference from which
                               # adaptive scaling is computed at every corpus size.


# ── Adaptive L_MAX ─────────────────────────────────────────────────────────────

def compute_l_max(n_docs: int) -> float:
    """
    Scale L_MAX linearly with corpus size.

    BitmapBrute latency is O(n_docs × selectivity).  The baseline value of
    100ms was calibrated at 200k documents; for larger corpora the ceiling
    must grow proportionally so the bitmap arm remains competitive up to the
    same target selectivity region.

    Example: 671 750 docs → 100 × (671 750 / 200 000) ≈ 336 ms
    """
    return CBO_L_MAX_BASE * max(1.0, n_docs / CBO_L_MAX_REF_DOCS)


def compute_n_warmup(n_docs: int) -> int:
    """
    Scale N_WARMUP linearly with corpus size.

    The number of observations needed to resolve the reward margin at the
    crossover grows as margin² shrinks.  The margin narrows as the corpus
    grows (adaptive L_MAX keeps bitmap L_norm roughly constant, but
    PostFilter SLA varies), so the convergence requirement scales with
    corpus size.

    The baseline of 5 000 is sufficient at 200k (delivers ~4.4× the
    required observations).  Scaling linearly ensures the same safety
    margin is maintained at larger scales.

    Examples:
        200 000 docs → 5 000  (baseline, unchanged)
        500 000 docs → 12 500
        671 750 docs → 16 794
    """
    return max(N_WARMUP_BASE, int(N_WARMUP_BASE * (n_docs / CBO_L_MAX_REF_DOCS)))
