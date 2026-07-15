# =============================================================================
# sweep_pesos_content.py — Varredura do split de pesos do Content-Based
#
# Avalia cada conjunto de pesos (CONFIGS) entre as features do content (artista/genero/ano) no mesmo
# conjunto de teste do compare_models.py, medindo Recall e NDCG@500.
#
# Por playlist, os vetores de score por feature são computados uma vez; cada config é só uma soma
# ponderada + top-k, no espaço do catálogo de conteúdo, mapeado para track_encoded para conferir os
# acertos.
# =============================================================================

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
from tqdm import tqdm

from data_processing import load_or_process_interactions
from compare_models import build_test_cases
from model_content import ContentRecommender

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

N_FILES            = 350
NUM_PLAYLISTS_TEST = 500
PCT_REMOVED        = 0.20
TOP_K              = 500
SEED               = 42

# Configs de pesos a testar (artista pesado; gênero/ano só complementam).
CONFIGS = [
    {'artista': 1.00},
    {'artista': 0.95, 'genero': 0.05},
    {'artista': 0.90, 'genero': 0.10},
    {'artista': 0.80, 'genero': 0.20},
    {'artista': 0.70, 'genero': 0.30},
    {'artista': 0.95, 'ano': 0.05},
    {'artista': 0.90, 'genero': 0.05, 'ano': 0.05},
    {'artista': 0.85, 'genero': 0.10, 'ano': 0.05},
]

# Todas as features que aparecem em algum config (computadas 1x por playlist)
FEATURES = sorted({f for c in CONFIGS for f in c})


def _nome_config(c):
    return ' / '.join(f"{k}={v:.2f}" for k, v in c.items())


if __name__ == "__main__":
    data = load_or_process_interactions(N_FILES)
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = data

    print(f"\nConjunto de teste (seed={SEED}, {NUM_PLAYLISTS_TEST} playlists, {PCT_REMOVED:.0%} removido)...")
    rng = np.random.default_rng(SEED)
    test_cases = build_test_cases(interactions_df, rng, NUM_PLAYLISTS_TEST, PCT_REMOVED)

    print("\nCarregando Content...")
    rec = ContentRecommender.load_or_train(*data)
    N = rec.N
    cidx_to_enc = rec.cidx_to_enc
    invalido_cat = cidx_to_enc < 0
    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    n_cfg = len(CONFIGS)
    hits        = np.zeros(n_cfg, dtype=np.int64)
    recall_sum  = np.zeros(n_cfg, dtype=np.float64)
    ndcg_sum    = np.zeros(n_cfg, dtype=np.float64)
    total_removed = 0

    t0 = time.time()
    for pid, removed, remaining in tqdm(test_cases, desc="Pesos", unit="pl"):
        removed_set = set(removed.tolist())
        n_rem = len(removed)
        total_removed += n_rem
        idcg = float(np.sum(1.0 / np.log2(np.arange(1, min(n_rem, TOP_K) + 1) + 1.0)))

        cidx = rec.enc_to_cidx[remaining]
        cidx = cidx[cidx >= 0]
        if len(cidx) == 0:
            continue

        feats = np.stack([rec._centroid_scores(cidx, {f: 1.0}) for f in FEATURES])
        bloq = invalido_cat.copy()
        bloq[cidx] = True

        for ci, cfg in enumerate(CONFIGS):
            total = sum(cfg.values()) or 1.0
            v = np.zeros(N, dtype=np.float32)
            for f, w in cfg.items():
                v += (w / total) * feats[feat_idx[f]]
            v[bloq] = -np.inf
            k = min(TOP_K, N)
            part = np.argpartition(v, -k)[-k:]
            order = part[np.argsort(v[part])[::-1]]
            order = order[np.isfinite(v[order])]
            enc = cidx_to_enc[order]

            hit_ranks = [r for r, e in enumerate(enc.tolist(), start=1) if e in removed_set]
            h = len(hit_ranks)
            hits[ci]       += h
            recall_sum[ci] += h / n_rem
            if hit_ranks:
                dcg = float(np.sum(1.0 / np.log2(np.asarray(hit_ranks) + 1.0)))
                ndcg_sum[ci] += dcg / idcg

    npl = len(test_cases)
    print(f"\nConcluído em {(time.time()-t0)/60:.1f} min ({npl} playlists, {total_removed} removidas)\n")

    recall_mi = hits / total_removed
    recall_ma = recall_sum / npl
    ndcg      = ndcg_sum / npl

    print("=" * 76)
    print(f"AFINAÇÃO DE PESOS DO CONTENT (Top-{TOP_K}, por Recall micro)")
    print("=" * 76)
    print(f"{'Config':<46} {'Recall(mi)':>9} {'Recall(ma)':>9} {'NDCG':>8}")
    print("-" * 76)
    for ci in np.argsort(-recall_mi):
        marca = "  <-" if ci == int(np.argmax(recall_mi)) else ""
        print(f"{_nome_config(CONFIGS[ci]):<46} {recall_mi[ci]:>9.2%} {recall_ma[ci]:>9.2%} {ndcg[ci]:>8.4f}{marca}")
    print("=" * 76)
    best = int(np.argmax(recall_mi))
    print(f"MELHOR: {_nome_config(CONFIGS[best])}  ->  Recall@{TOP_K} = {recall_mi[best]:.2%}")
