# =============================================================================
# sweep_hybrid_alpha.py — Varredura do peso ALPHA do híbrido (late fusion)
#
# Mede, no mesmo conjunto de teste do compare_models.py, como o Recall/NDCG do
# híbrido variam com ALPHA (peso do colaborativo; 1-ALPHA vai para o content).
#
# O score do colaborativo e a normalização min-max são computados uma vez por
# playlist; para cada ALPHA só se refaz a combinação ponderada + top-k. ALPHA=1.0
# equivale ao colaborativo puro e ALPHA=0.0 ao content puro.
#
# COLAB_KEY define o parceiro colaborativo ("neumf" | "itemknn" | "als").
# =============================================================================

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
from tqdm import tqdm

from data_processing import load_or_process_interactions
from compare_models import build_test_cases  # MESMO conjunto de teste
from model_hybrid import COLAB_REGISTRY, _minmax

# =============================================================================
# CONFIGURAÇÃO — espelha o compare_models.py (comparação justa)
# =============================================================================

N_FILES            = 350
NUM_PLAYLISTS_TEST = 500
PCT_REMOVED        = 0.20
TOP_K              = 500
SEED               = 42

COLAB_KEY = "neumf"   # "neumf" | "itemknn" | "als"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]


def _fuse_topk(nc, pc, nb, pb, alpha, num_tracks, top_k):
    """Combinação ponderada (igual ao HybridRecommender) para um ALPHA dado."""
    wc, wb = alpha, 1.0 - alpha
    num = wc * nc * pc + wb * nb * pb
    den = wc * pc + wb * pb
    scores = np.full(num_tracks, -np.inf, dtype=np.float64)
    ok = den > 0
    scores[ok] = num[ok] / den[ok]
    k = min(top_k, num_tracks)
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(scores[part])[::-1]]


if __name__ == "__main__":
    data = load_or_process_interactions(N_FILES)
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = data

    print(f"\nConjunto de teste (seed={SEED}, {NUM_PLAYLISTS_TEST} playlists, "
          f"{PCT_REMOVED:.0%} removido)...")
    rng = np.random.default_rng(SEED)
    test_cases = build_test_cases(interactions_df, rng,
                                  NUM_PLAYLISTS_TEST, PCT_REMOVED)

    # ---- Carrega os DOIS componentes ----
    mod_name, cls_name, _ = COLAB_REGISTRY[COLAB_KEY]
    import importlib
    ColabCls = getattr(importlib.import_module(mod_name), cls_name)
    from model_content import ContentRecommender
    print(f"\nCarregando {cls_name} + Content (par: {COLAB_KEY}+content)...")
    colab   = ColabCls.load_or_train(*data)
    content = ContentRecommender.load_or_train(*data)
    num_tracks = colab.num_tracks

    # ---- Acumuladores por ALPHA ----
    n_alpha = len(ALPHAS)
    hits        = np.zeros(n_alpha, dtype=np.int64)   # p/ recall micro
    recall_sum  = np.zeros(n_alpha, dtype=np.float64) # p/ recall macro
    ndcg_sum    = np.zeros(n_alpha, dtype=np.float64)
    total_removed = 0

    t0 = time.time()
    for pid, removed, remaining in tqdm(test_cases, desc=f"Varredura {COLAB_KEY}",
                                        unit="pl"):
        removed_set = set(removed.tolist())
        n_rem = len(removed)
        total_removed += n_rem

        # Caro: score dos dois modelos UMA vez por playlist
        s_c = np.asarray(colab.score(pid_map[pid], remaining),   dtype=np.float64)
        s_b = np.asarray(content.score(pid_map[pid], remaining), dtype=np.float64)
        # Exclui as conhecidas ANTES de normalizar (escala = só candidatas)
        s_c[remaining] = -np.inf
        s_b[remaining] = -np.inf
        nc, pc = _minmax(s_c)
        nb, pb = _minmax(s_b)

        idcg = float(np.sum(1.0 / np.log2(np.arange(1, min(n_rem, TOP_K) + 1) + 1.0)))

        for ai, alpha in enumerate(ALPHAS):
            top = _fuse_topk(nc, pc, nb, pb, alpha, num_tracks, TOP_K)
            hit_ranks = [r for r, t in enumerate(top.tolist(), start=1)
                         if t in removed_set]
            h = len(hit_ranks)
            hits[ai]       += h
            recall_sum[ai] += h / n_rem
            if hit_ranks:
                dcg = float(np.sum(1.0 / np.log2(np.asarray(hit_ranks) + 1.0)))
                ndcg_sum[ai] += dcg / idcg

    npl = len(test_cases)
    print(f"\nVarredura concluída em {(time.time()-t0)/60:.1f} min "
          f"({npl} playlists, {total_removed} removidas)\n")

    # ---- Tabela ----
    print("=" * 64)
    print(f"VARREDURA DE ALPHA — Híbrido {COLAB_KEY}+content (Top-{TOP_K})")
    print(f"ALPHA = peso do colaborativo (1-ALPHA = peso do content)")
    print("=" * 64)
    head = f"{'ALPHA':>6} {'Recall(mi)':>11} {'Recall(ma)':>11} {'NDCG':>9}"
    print(head)
    print("-" * len(head))
    best_ai = int(np.argmax(hits))
    for ai, alpha in enumerate(ALPHAS):
        marca = "  <- melhor" if ai == best_ai else ""
        tag = ("  (=content puro)" if alpha == 0.0 else
               "  (=colab puro)"  if alpha == 1.0 else "")
        print(f"{alpha:>6.2f} {hits[ai]/total_removed:>10.2%} "
              f"{recall_sum[ai]/npl:>10.2%} {ndcg_sum[ai]/npl:>9.4f}{marca}{tag}")
    print("=" * 64)
    print(f"Melhor ALPHA por Recall(micro): {ALPHAS[best_ai]:.2f} "
          f"({hits[best_ai]/total_removed:.2%})")
