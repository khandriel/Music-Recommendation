# =============================================================================
# compare_models.py — Avaliação e comparação dos modelos de recomendação
#
# Toda a LÓGICA DE TESTE vive aqui (os arquivos de modelo apenas treinam):
#
#   1. Gera, com SEED fixa, o conjunto de teste: QUAIS playlists são escolhidas
#      e QUAIS músicas de cada uma são removidas. O conjunto é gerado UMA única
#      vez e reutilizado por todos os modelos — comparação justa por construção.
#   2. Verifica quais modelos existem em saved_models/ (este script NÃO treina
#      nada: modelos ausentes são pulados — treine com main.py).
#   3. Avalia cada modelo disponível, UM DE CADA VEZ (sequencial, liberando a
#      memória entre eles). A falha de um não derruba os demais.
#   4. Calcula Recall@K (micro e macro), Precision@K, F1@K e NDCG@K e grava um
#      relatório TXT com a configuração, o conjunto de teste e os resultados.
#
# MODELOS COMPARADOS:
#   - ALS, Item-kNN, NeuMF  -> colaborativos (co-ocorrência em playlists)
#   - Content-Based         -> PURO conteúdo (áudio + gênero + ano); NÃO usa
#                              co-ocorrência. Só pontua faixas que têm features
#                              de áudio (audio_features.csv); as demais ficam de
#                              fora do ranking — a cobertura parcial é uma
#                              característica honesta do modelo, não é corrigida.
#   - Híbrido NeuMF+Content -> late fusion (model_hybrid.py): combina os scores
#                              normalizados do NeuMF e do content. Só aparece se
#                              AMBOS os componentes estiverem treinados.
#
# FLAGS:
#   SHOW_DETAILS — inclui no TXT as playlists escolhidas e as músicas removidas
#                  de cada uma (o relatório fica grande; só vai para o arquivo,
#                  não para o console).
#
# USO: python compare_models.py
# =============================================================================

import os
# BLAS em 1 thread ANTES de qualquer import que carregue o numpy — evita
# oversubscription com o paralelismo interno do implicit (mesmo ajuste feito
# nos arquivos de modelo, repetido aqui porque o numpy entra antes deles).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import sys
import time
import traceback
from datetime import datetime

import numpy as np
from tqdm import tqdm

from data_processing import load_or_process_interactions

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

N_FILES            = 350    # mesmo valor do main.py (compartilha o cache/)
NUM_PLAYLISTS_TEST = 500    # playlists de teste — as MESMAS para todos os modelos
PCT_REMOVED        = 0.20   # fração de músicas removida de cada playlist
TOP_K              = 500    # tamanho da lista de recomendação avaliada
SEED               = 42     # fixa playlists escolhidas + músicas removidas
SHOW_DETAILS       = False  # True = inclui playlists/músicas removidas no TXT
REPORT_PATH        = "comparison_report.txt"

MODELS_DIR = "saved_models"

# =============================================================================
# REGISTRO DOS MODELOS
# Cada entrada: (nome, caminho do modelo salvo, fábrica que carrega o modelo).
# O import fica dentro da fábrica: TensorFlow/implicit/conteúdo só carregam na
# vez do respectivo modelo.
# =============================================================================

def _make_als(data):
    from model_collaborative_als import ALSRecommender
    return ALSRecommender.load_or_train(*data)


def _make_itemknn(data):
    from model_collaborative_itemknn import ItemKNNRecommender
    return ItemKNNRecommender.load_or_train(*data)


def _make_neumf(data):
    from model_collaborative_neumf import NeuMFRecommender
    return NeuMFRecommender.load_or_train(*data)


def _make_content(data):
    from model_content import ContentRecommender
    return ContentRecommender.load_or_train(*data)


def _make_hybrid_neumf(data):
    from model_hybrid import HybridRecommender
    return HybridRecommender.build(data, colab_key="neumf")


# O caminho de cada modelo pode ser uma str (1 arquivo) ou uma lista (o híbrido
# compõe dois modelos salvos; só fica disponível se TODOS existirem).
from model_hybrid import required_paths as _hybrid_paths

MODELS = [
    ("ALS (Matrix Factorization)", os.path.join(MODELS_DIR, "collaborative_als.npz"),     _make_als),
    ("Item-kNN (co-ocorrência)",   os.path.join(MODELS_DIR, "collaborative_itemknn.npz"), _make_itemknn),
    ("Content-Based (áudio+gênero+ano)", os.path.join(MODELS_DIR, "content_based.joblib"), _make_content),
    ("NeuMF (rede neural)",        os.path.join(MODELS_DIR, "collaborative_neumf.keras"), _make_neumf),
    ("Híbrido NeuMF+Content (late fusion)", _hybrid_paths("neumf"), _make_hybrid_neumf),
]


def _model_available(path) -> bool:
    """True se o(s) arquivo(s) requerido(s) existe(m) — str ou lista de str."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    return all(os.path.exists(p) for p in paths)


def _fmt_path(path) -> str:
    return " + ".join(path) if isinstance(path, (list, tuple)) else path

# =============================================================================
# CONJUNTO DE TESTE — gerado UMA vez, compartilhado por todos os modelos
# =============================================================================

def build_test_cases(interactions_df, rng, num_playlists_test, pct_removed):
    """
    Sorteia `num_playlists_test` playlists e, de cada uma, remove
    `pct_removed` das músicas (no mínimo 1). Retorna uma lista de casos
    (pid, removed, remaining), onde removed/remaining são arrays de
    track_encoded. Como o conjunto é materializado aqui e reutilizado,
    todos os modelos são avaliados sobre EXATAMENTE os mesmos dados.
    """
    all_pids = interactions_df['pid'].unique()
    n_test   = min(num_playlists_test, len(all_pids))
    chosen   = rng.choice(all_pids, size=n_test, replace=False)

    # Índice pid → tracks restrito às playlists sorteadas, em UMA passada
    # (filtrar o DataFrame inteiro a cada playlist dominaria o tempo).
    sub = interactions_df[interactions_df['pid'].isin(set(chosen.tolist()))]
    tracks_by_pid = {pid: g.values for pid, g in sub.groupby('pid')['track_encoded']}

    cases = []
    for pid in chosen:
        tracks      = tracks_by_pid[pid]
        n_rem       = max(1, int(len(tracks) * pct_removed))
        removed     = rng.choice(tracks, n_rem, replace=False)
        removed_set = set(removed.tolist())
        remaining   = np.array(
            [t for t in tracks if t not in removed_set], dtype=np.int64)
        cases.append((int(pid), removed, remaining))
    return cases

# =============================================================================
# AVALIAÇÃO — métricas de ranking sobre o conjunto de teste compartilhado
# =============================================================================

def evaluate_model(nome, rec, test_cases, pid_map, top_k):
    """
    Para cada caso de teste, pede o Top-K ao modelo a partir das músicas
    RESTANTES (excluindo-as do ranking) e mede quantas das REMOVIDAS foram
    recuperadas. Retorna um dicionário de métricas:

      recall_micro — total de acertos / total de removidas (visão global)
      recall_macro — média do recall por playlist (cada playlist pesa igual)
      precision    — acertos / K, média por playlist
      f1           — média harmônica de precision e recall, média por playlist
      ndcg         — qualidade do RANKING: acertos no topo valem mais
                     (DCG/IDCG, média por playlist)
    """
    recalls, precisions, f1s, ndcgs = [], [], [], []
    total_hits = total_removed = 0

    for pid, removed, remaining in tqdm(test_cases, desc=f"Avaliando {nome}",
                                        unit="pl"):
        removed_set = set(removed.tolist())
        top = rec.recommend(pid_map[pid], remaining, remaining, top_k)

        hit_ranks = [r for r, t in enumerate(top, start=1) if t in removed_set]
        h, n_rem  = len(hit_ranks), len(removed)

        recall    = h / n_rem
        precision = h / top_k
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        dcg  = float(np.sum(1.0 / np.log2(np.asarray(hit_ranks) + 1.0))) if hit_ranks else 0.0
        idcg = float(np.sum(1.0 / np.log2(np.arange(1, min(n_rem, top_k) + 1) + 1.0)))

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        ndcgs.append(dcg / idcg)
        total_hits    += h
        total_removed += n_rem

    return {
        'recall_micro': total_hits / total_removed,
        'recall_macro': float(np.mean(recalls)),
        'precision':    float(np.mean(precisions)),
        'f1':           float(np.mean(f1s)),
        'ndcg':         float(np.mean(ndcgs)),
        'hits':         total_hits,
        'removed':      total_removed,
    }

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    data = load_or_process_interactions(N_FILES)
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = data

    # ------------------------------------------------------------------ #
    # 1. Conjunto de teste (uma única vez, com a SEED)
    # ------------------------------------------------------------------ #
    print(f"\nGerando conjunto de teste (seed={SEED}, "
          f"{NUM_PLAYLISTS_TEST} playlists, {PCT_REMOVED:.0%} removido)...")
    rng = np.random.default_rng(SEED)
    test_cases = build_test_cases(interactions_df, rng,
                                  NUM_PLAYLISTS_TEST, PCT_REMOVED)
    total_removidas = sum(len(removed) for _, removed, _ in test_cases)

    # ------------------------------------------------------------------ #
    # 2. Cabeçalho do relatório + verificação dos modelos disponíveis
    # ------------------------------------------------------------------ #
    rep = []
    rep.append("=" * 70)
    rep.append("COMPARAÇÃO DOS MODELOS DE RECOMENDAÇÃO — RELATÓRIO")
    rep.append(f"Gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}")
    rep.append("=" * 70)
    rep.append("")
    rep.append("CONFIGURAÇÃO")
    rep.append(f"  N_FILES (arquivos do dataset) : {N_FILES}")
    rep.append(f"  Playlists de teste            : {len(test_cases)}")
    rep.append(f"  Fração removida por playlist  : {PCT_REMOVED:.0%}")
    rep.append(f"  Top-K avaliado                : {TOP_K}")
    rep.append(f"  Seed                          : {SEED}")
    rep.append("")
    rep.append("DATASET")
    rep.append(f"  Playlists      : {len(pid_map)}")
    rep.append(f"  Músicas únicas : {len(track_map)}")
    rep.append(f"  Interações     : {len(interactions_df)}")
    rep.append("")
    rep.append("MODELOS")

    disponiveis = []
    for nome, path, factory in MODELS:
        if _model_available(path):
            rep.append(f"  [OK]      {nome} — '{_fmt_path(path)}'")
            disponiveis.append((nome, factory))
        else:
            rep.append(f"  [AUSENTE] {nome} — '{_fmt_path(path)}' não encontrado; "
                       f"treine com main.py (modelo PULADO)")
    rep.append("")

    for linha in rep:
        print(linha)

    # ------------------------------------------------------------------ #
    # 3. Conjunto de teste no relatório (detalhe controlado por flag)
    # ------------------------------------------------------------------ #
    rep.append(f"CONJUNTO DE TESTE: {len(test_cases)} playlists | "
               f"{total_removidas} músicas removidas no total")
    if SHOW_DETAILS:
        rep.append("-" * 70)
        for pid, removed, remaining in test_cases:
            rep.append(f"Playlist {pid} — {len(removed) + len(remaining)} músicas "
                       f"({len(removed)} removidas, {len(remaining)} restantes)")
            for idx in removed:
                uri = reverse_track_map[int(idx)]
                rep.append(f"    - {uri_to_name.get(uri, '?')}")
        rep.append("-" * 70)
    else:
        rep.append("(playlists escolhidas e músicas removidas ocultas — "
                   "use SHOW_DETAILS=True para incluí-las)")
    rep.append("")

    # ------------------------------------------------------------------ #
    # 4. Avaliação sequencial dos modelos disponíveis
    # ------------------------------------------------------------------ #
    resultados = {}
    for nome, factory in disponiveis:
        print("\n" + "#" * 70)
        print(f"# {nome}")
        print("#" * 70)
        t0 = time.time()
        try:
            rec = factory(data)
            metricas = evaluate_model(nome, rec, test_cases, pid_map, TOP_K)
            metricas['tempo_min'] = (time.time() - t0) / 60
            resultados[nome] = metricas
            print(f"[{nome}] Recall@{TOP_K} (micro) = "
                  f"{metricas['recall_micro']:.2%} | NDCG@{TOP_K} = "
                  f"{metricas['ndcg']:.4f}")
            del rec
        except Exception:
            print(f"\n*** '{nome}' falhou — seguindo para o próximo ***")
            traceback.print_exc()
            resultados[nome] = None
        # Se o NeuMF carregou o TensorFlow, libera a sessão antes do próximo
        if 'keras' in sys.modules:
            try:
                sys.modules['keras'].backend.clear_session()
            except Exception:
                pass
        gc.collect()

    # ------------------------------------------------------------------ #
    # 5. Tabela final (console + relatório) e gravação do TXT
    # ------------------------------------------------------------------ #
    tab = []
    tab.append("=" * 70)
    tab.append(f"RESULTADOS — {len(test_cases)} playlists, "
               f"{total_removidas} músicas removidas, Top-{TOP_K}")
    tab.append("=" * 70)
    tab.append("Métricas (médias por playlist, exceto o recall micro):")
    tab.append(f"  Recall@{TOP_K} (micro) : removidas recuperadas / total de removidas")
    tab.append(f"  Recall@{TOP_K} (macro) : média do recall por playlist")
    tab.append(f"  Precision@{TOP_K}      : acertos / {TOP_K}")
    tab.append(f"  F1@{TOP_K}             : média harmônica de precision e recall")
    tab.append(f"  NDCG@{TOP_K}           : qualidade do ranking (acertos no topo valem mais)")
    tab.append("")
    header = (f"{'Modelo':<35} {'Recall(mi)':>10} {'Recall(ma)':>10} "
              f"{'Precision':>10} {'F1':>8} {'NDCG':>8} {'Tempo':>8}")
    tab.append(header)
    tab.append("-" * len(header))
    for nome, path, _ in MODELS:
        m = resultados.get(nome)
        if m is not None:
            tab.append(f"{nome:<35} {m['recall_micro']:>10.2%} "
                       f"{m['recall_macro']:>10.2%} {m['precision']:>10.2%} "
                       f"{m['f1']:>8.2%} {m['ndcg']:>8.4f} "
                       f"{m['tempo_min']:>6.1f}m")
        elif nome in resultados:
            tab.append(f"{nome:<35} FALHOU (ver console/traceback)")
        else:
            tab.append(f"{nome:<35} não treinado — pulado")
    tab.append("=" * 70)
    tab.append("Obs.: o conjunto de teste é gerado uma única vez e reutilizado por")
    tab.append("todos os modelos (mesmas playlists, mesmas músicas removidas).")
    tab.append("NeuMF representa a playlist pelo embedding aprendido no treino;")
    tab.append("ALS e Item-kNN apenas pelas músicas restantes (fold-in).")
    tab.append("Content-Based usa SÓ conteúdo (áudio+gênero+ano): a playlist vira")
    tab.append("o perfil-centroide das restantes; faixas sem features de áudio não")
    tab.append("entram no ranking (cobertura parcial, reportada como está).")
    tab.append("Híbrido = late fusion (min-max + média ponderada, alpha no colab):")
    tab.append("combina os scores normalizados de NeuMF e Content; faixa sem áudio")
    tab.append("usa só o colaborativo (o content se abstém, sem penalizá-la).")

    rep.extend(tab)
    for linha in tab:
        print(linha)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(rep) + "\n")
    print(f"\nRelatório completo salvo em '{REPORT_PATH}'")
