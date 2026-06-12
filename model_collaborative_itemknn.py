# =============================================================================
# model_collaborative_itemknn.py — Filtragem Colaborativa pura via item-item
# kNN por co-ocorrência
#
# ARQUIVO AUTOCONTIDO: este arquivo só TREINA o modelo e expõe a inferência
# (recommend). A lógica de teste/avaliação — escolha das playlists, remoção
# de músicas e métricas — vive em compare_collaborative.py, que usa o MESMO
# conjunto de teste para todos os modelos (comparação justa).
#
# Modelo EXCLUSIVAMENTE colaborativo e BASEADO EM MEMÓRIA (não-paramétrico):
# para cada música, guarda os K vizinhos mais parecidos — similaridade do
# cosseno entre as colunas da matriz playlist×música (músicas que co-ocorrem
# nas mesmas playlists). Recomendar para uma playlist = somar a similaridade
# de cada música restante a todas as candidatas:
#
#     scores = (vetor binário das músicas restantes) · S
#
# onde S (músicas × músicas) é a matriz de similaridade podada nos top-K. Como
# S é esparsa, o produto é um matvec esparso barato. A poda em K vizinhos
# remove ruído de co-ocorrências fracas.
#
# DIFERENÇA PARA O ALS: o kNN só liga músicas que REALMENTE co-ocorreram
# (relação direta); o ALS, por aprender fatores latentes, capta relações
# indiretas (músicas que combinam mesmo sem nunca terem aparecido juntas).
#
# DECISÃO DE PROJETO — treino com `implicit`, inferência manual: a lib é usada
# só para calcular a matriz de similaridade (rápido, em Cython); o score é um
# produto esparso feito à mão, sem depender da assinatura instável de
# model.recommend() entre versões do implicit.
#
# USO:
#   Treinar:   python model_collaborative_itemknn.py   (ou via main.py)
#   Comparar:  python compare_collaborative.py
# =============================================================================

import os
# `implicit` paraleliza internamente; deixar o BLAS abrir o próprio threadpool
# degrada a performance (oversubscription). Fixar ANTES de importar numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import sparse

from data_processing import load_or_process_interactions

# =============================================================================
# PERSISTÊNCIA E HIPERPARÂMETROS
# =============================================================================

MODELS_DIR = "saved_models"
MODEL_PATH = os.path.join(MODELS_DIR, "collaborative_itemknn.npz")

KNN_K = 100  # Vizinhos por música guardados na matriz de similaridade


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

# =============================================================================
# MODELO
# =============================================================================

class ItemKNNRecommender:
    """Item-item kNN por co-ocorrência (cosseno podado nos K vizinhos)."""

    def __init__(self, interactions_df, pid_map, track_map):
        self.interactions_df = interactions_df
        self.num_playlists   = len(pid_map)
        self.num_tracks      = len(track_map)
        self.model           = None

    # --------------------------------------------------------------------- #
    # FACTORY
    # --------------------------------------------------------------------- #
    @classmethod
    def load_or_train(cls, interactions_df, pid_map, track_map,
                      reverse_track_map=None, uri_to_name=None):
        """
        Assinatura idêntica nos três modelos colaborativos (drop-in no main.py
        e no compare_collaborative.py). reverse_track_map e uri_to_name não
        são usados aqui — nomes de músicas só importam no relatório do compare.
        """
        self = cls(interactions_df, pid_map, track_map)
        if model_exists():
            print(f"[Item-kNN] Modelo encontrado — carregando de '{MODEL_PATH}'...")
            self.load()
        else:
            print("[Item-kNN] Nenhum modelo salvo — treinando do zero...")
            self.train()
            self.save()
            print(f"[Item-kNN] Modelo salvo em '{MODEL_PATH}'")
        return self

    # --------------------------------------------------------------------- #
    # TREINO
    # --------------------------------------------------------------------- #
    def _build_user_items(self) -> sparse.csr_matrix:
        """Matriz esparsa binária playlist×música (implicit feedback)."""
        rows = self.interactions_df['pid_encoded'].to_numpy()
        cols = self.interactions_df['track_encoded'].to_numpy()
        data = np.ones(len(rows), dtype=np.float32)
        ui = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(self.num_playlists, self.num_tracks),
        )
        ui.data[:] = 1.0   # csr soma duplicatas → força tudo a binário
        return ui

    def train(self):
        from implicit.nearest_neighbours import CosineRecommender

        print(f"[Item-kNN] Treinando (K={KNN_K}, similaridade do cosseno)...")
        model = CosineRecommender(K=KNN_K)
        model.fit(self._build_user_items())   # (playlists × músicas)

        self.model = model
        self.similarity = model.similarity.tocsr()  # (músicas × músicas), esparsa

    # --------------------------------------------------------------------- #
    # PERSISTÊNCIA
    # --------------------------------------------------------------------- #
    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.model.save(MODEL_PATH)

    def load(self):
        from implicit.nearest_neighbours import CosineRecommender
        self.model = CosineRecommender.load(MODEL_PATH)
        self.similarity = self.model.similarity.tocsr()

    # --------------------------------------------------------------------- #
    # INFERÊNCIA
    # --------------------------------------------------------------------- #
    def recommend(self, pid_encoded, seed_idxs, exclude_idxs, top_k=500):
        """
        Top-k músicas somando a similaridade das músicas em `seed_idxs`
        (músicas conhecidas da playlist) a todas as candidatas, excluindo
        `exclude_idxs`. `pid_encoded` é IGNORADO — a interface é a mesma nos
        três modelos colaborativos; aqui a playlist é representada apenas
        pelas músicas seed.
        """
        seed_idxs = np.asarray(seed_idxs, dtype=np.int64)
        if len(seed_idxs) == 0:
            scores = np.zeros(self.num_tracks, dtype=np.float64)
        else:
            # Vetor-linha binário das músicas seed (1 × num_tracks)
            row = sparse.csr_matrix(
                (np.ones(len(seed_idxs), dtype=np.float64),
                 (np.zeros(len(seed_idxs), dtype=np.int64), seed_idxs)),
                shape=(1, self.num_tracks),
            )
            scores = np.asarray((row @ self.similarity).todense()).ravel()

        exclude_idxs = np.asarray(exclude_idxs, dtype=np.int64)
        if len(exclude_idxs) > 0:
            scores[exclude_idxs] = -np.inf

        k = min(top_k, scores.shape[0])
        # argpartition: separa os k maiores sem ordenar o vetor inteiro
        part = np.argpartition(scores, -k)[-k:]
        return part[np.argsort(scores[part])[::-1]].tolist()


# =============================================================================
# EXECUÇÃO DIRETA — apenas TREINA (avaliação: compare_collaborative.py)
# =============================================================================

if __name__ == "__main__":
    data = load_or_process_interactions()
    ItemKNNRecommender.load_or_train(*data)
    print("[Item-kNN] Concluído. Para avaliar/comparar: python compare_collaborative.py")
