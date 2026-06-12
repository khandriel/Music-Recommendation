# =============================================================================
# main.py — Ponto de entrada de TREINO dos modelos colaborativos
#
# Escolha nas flags abaixo QUAIS modelos treinar. Os escolhidos rodam UM DE
# CADA VEZ (sequencial — pensado para máquinas que não comportam mais de um
# modelo na memória). Para treinar todos, deixe todas as flags em True.
#
# Modelos já salvos em saved_models/ NÃO são retreinados — são apenas
# carregados (a mensagem indica qual caso ocorreu). Para forçar o retreino,
# apague o arquivo correspondente em saved_models/.
#
# A AVALIAÇÃO/COMPARAÇÃO não acontece aqui: toda a lógica de teste (escolha
# das playlists, remoção de músicas, métricas, relatório) vive em
# compare_collaborative.py.
#
# USO:
#   Treinar:   python main.py
#   Comparar:  python compare_collaborative.py
#
# NOTA: o content-based (model_content.py) e o híbrido (model_hybrid.py)
# estão DESATIVADOS por enquanto (dependem do model_content, fora do ar).
# A fiação antiga deles está no histórico do git (main.py anterior).
# =============================================================================

# =============================================================================
# CONFIGURAÇÃO — escolha o que treinar
# =============================================================================
TRAIN_ALS     = True   # Matrix Factorization implícita (leve)
TRAIN_ITEMKNN = True   # Item-item kNN por co-ocorrência (leve)
TRAIN_NEUMF   = True   # Rede neural NeuMF (PESADO — fica por último)

N_FILES = 350          # Arquivos JSON do dataset (cada um = 1000 playlists)

# =============================================================================
# EXECUÇÃO
# =============================================================================

import os
# BLAS em 1 thread ANTES de qualquer import que carregue o numpy — evita
# oversubscription com o paralelismo interno do implicit (mesmo ajuste feito
# nos arquivos de modelo, repetido aqui porque o numpy entra antes deles).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import time
import traceback

from data_processing import load_or_process_interactions


# Os imports dos modelos ficam DENTRO das funções: TensorFlow/implicit só são
# carregados na vez do respectivo modelo (não pesam na memória dos demais).

def treinar_als(data):
    from model_collaborative_als import ALSRecommender
    ALSRecommender.load_or_train(*data)


def treinar_itemknn(data):
    from model_collaborative_itemknn import ItemKNNRecommender
    ItemKNNRecommender.load_or_train(*data)


def treinar_neumf(data):
    from model_collaborative_neumf import NeuMFRecommender
    NeuMFRecommender.load_or_train(*data)
    import keras
    keras.backend.clear_session()   # libera o grafo TF após salvar


# Ordem: do mais leve ao mais pesado (NeuMF/TensorFlow por último)
PLANO = [
    ("ALS (Matrix Factorization)", TRAIN_ALS,     treinar_als),
    ("Item-kNN (co-ocorrência)",   TRAIN_ITEMKNN, treinar_itemknn),
    ("NeuMF (rede neural)",        TRAIN_NEUMF,   treinar_neumf),
]

if __name__ == "__main__":
    data = load_or_process_interactions(N_FILES)

    resultados = []
    for nome, ativo, fn in PLANO:
        if not ativo:
            resultados.append((nome, "pulado (flag desativada)"))
            continue
        print("\n" + "#" * 64)
        print(f"# TREINO: {nome}")
        print("#" * 64)
        t0 = time.time()
        try:
            fn(data)
            status = f"ok ({(time.time() - t0) / 60:.1f} min)"
        except Exception:
            print(f"\n*** '{nome}' falhou — seguindo para o próximo ***")
            traceback.print_exc()
            status = "FALHOU"
        resultados.append((nome, status))
        gc.collect()   # libera a memória do modelo antes do próximo

    print("\n" + "=" * 64)
    print("RESUMO DO TREINO")
    print("=" * 64)
    for nome, status in resultados:
        print(f"  {nome:<30} {status}")
    print("=" * 64)
    print("Para avaliar e comparar os modelos: python compare_collaborative.py")
