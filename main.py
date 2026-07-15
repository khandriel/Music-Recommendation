# =============================================================================
# main.py — ponto de entrada de TREINO dos modelos de recomendação
#
# Ligue nas flags abaixo quais modelos treinar. Os escolhidos rodam um de cada vez (sequencial,
# pensado para máquinas que não seguram vários modelos na memória ao mesmo tempo). Modelos já salvos
# em saved_models/ são só carregados, não retreinados — para forçar o retreino, apague o arquivo
# correspondente.
#
# A avaliação/comparação NÃO acontece aqui — isso vive em compare_models.py.
#
#   Treinar:   python main.py
#   Comparar:  python compare_models.py
# =============================================================================

# Escolha o que treinar
TRAIN_ALS     = False   # Matrix Factorization implícita (leve)
TRAIN_ITEMKNN = False   # Item-item kNN por co-ocorrência (leve)
TRAIN_CONTENT = False   # Content-Based puro (áudio+gênero+ano) a partir do CSV
TRAIN_NEUMF   = False   # Rede neural NeuMF (PESADO — fica por último)
TRAIN_HYBRID  = False   # Híbrido NeuMF+Content (compõe; treina o que faltar)

N_FILES = 350   # Arquivos JSON do dataset (cada um = 1000 playlists)

import os
# BLAS em 1 thread antes de qualquer import que puxe o numpy, senão o implicit sofre
# oversubscription. Os arquivos de modelo repetem esse ajuste.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc
import time
import traceback

from data_processing import load_or_process_interactions


# Os imports dos modelos ficam dentro das funções: TensorFlow/implicit só são carregados na vez do
# respectivo modelo, sem pesar na memória dos demais.

def treinar_als(data):
    from model_collaborative_als import ALSRecommender
    ALSRecommender.load_or_train(*data)


def treinar_itemknn(data):
    from model_collaborative_itemknn import ItemKNNRecommender
    ItemKNNRecommender.load_or_train(*data)


def treinar_content(data):
    from model_content import ContentRecommender
    ContentRecommender.load_or_train(*data)


def treinar_neumf(data):
    from model_collaborative_neumf import NeuMFRecommender
    NeuMFRecommender.load_or_train(*data)
    import keras
    keras.backend.clear_session()   # libera o grafo TF após salvar


def treinar_hybrid(data):
    # Late fusion: não treina nada próprio — compõe NeuMF + Content, treinando os componentes que
    # ainda não existirem em saved_models/.
    from model_hybrid import HybridRecommender
    HybridRecommender.load_or_train(*data)
    try:
        import keras
        keras.backend.clear_session()
    except Exception:
        pass


# Ordem do mais leve ao mais pesado: NeuMF/TensorFlow por último, e o híbrido depois dele, já que
# reaproveita o NeuMF e o content treinados.
PLANO = [
    ("ALS (Matrix Factorization)",       TRAIN_ALS,     treinar_als),
    ("Item-kNN (co-ocorrência)",         TRAIN_ITEMKNN, treinar_itemknn),
    ("Content-Based (áudio+gênero+ano)", TRAIN_CONTENT, treinar_content),
    ("NeuMF (rede neural)",              TRAIN_NEUMF,   treinar_neumf),
    ("Híbrido NeuMF+Content",            TRAIN_HYBRID,  treinar_hybrid),
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
    print("Para avaliar e comparar os modelos: python compare_models.py")
