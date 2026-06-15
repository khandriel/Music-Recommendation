# =============================================================================
# model_collaborative_als.py — Filtragem Colaborativa pura via Matrix
# Factorization implícita (Alternating Least Squares, Hu/Koren 2008)
#
# ARQUIVO AUTOCONTIDO: este arquivo só TREINA o modelo e expõe a inferência
# (recommend). A lógica de teste/avaliação — escolha das playlists, remoção
# de músicas e métricas — vive em compare_models.py, que usa o MESMO
# conjunto de teste para todos os modelos (comparação justa).
#
# Modelo EXCLUSIVAMENTE colaborativo: aprende fatores latentes P (playlists×f)
# e Q (músicas×f) tais que P·Qᵀ aproxima a matriz esparsa de interações
# playlist×música, ponderada por confiança Cui = 1 + α·r. O score de uma
# música para uma playlist é o produto escalar dos vetores latentes. Por
# operar sobre a matriz esparsa, é MUITO mais leve em memória que o NeuMF.
#
# INFERÊNCIA (fold-in): na avaliação, a playlist de teste só "mostra" suas
# músicas RESTANTES. O vetor latente u da playlist sai em forma fechada do
# mesmo problema de mínimos quadrados que o ALS resolve por linha:
#
#     u = (QᵀCQ + λI)⁻¹ Qᵀ C p
#
# Com Cui = 1 + α·r e preferência p = 1 nas restantes (0 caso contrário):
#     QᵀCQ  = QᵀQ + α·Qrᵀ·Qr        (Qr = fatores das músicas restantes)
#     QᵀC p = (1 + α)·Σ linhas de Qr
# QᵀQ é pré-computado uma vez; o sistema é só f×f → resolução instantânea.
# Os scores de todas as músicas são então Q·u.
#
# DECISÃO DE PROJETO — treino com `implicit`, inferência manual: a lib é usada
# só para TREINAR (a parte pesada, em Cython); a inferência manual evita a
# assinatura instável de model.recommend() entre versões do implicit e deixa
# a matemática explícita (útil para o TCC).
#
# USO:
#   Treinar:   python model_collaborative_als.py   (ou via main.py)
#   Comparar:  python compare_models.py
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
MODEL_PATH = os.path.join(MODELS_DIR, "collaborative_als.npz")

SEED           = 42    # Reprodutibilidade do treino
ALS_FACTORS    = 64    # Dimensão dos fatores latentes (equivale ao EMBEDDING_DIM do NeuMF)
ALS_REG        = 0.05  # Regularização L2 dos fatores
ALS_ALPHA      = 40    # Peso de confiança das interações (Cui = 1 + alpha·r)
ALS_ITERATIONS = 20    # Iterações de ALS (cada uma resolve playlists e músicas)


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

# =============================================================================
# MODELO
# =============================================================================

class ALSRecommender:
    """Matrix Factorization implícita (ALS) — colaborativo puro, esparso."""

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
        e no compare_models.py). reverse_track_map e uri_to_name não
        são usados aqui — nomes de músicas só importam no relatório do compare.
        """
        self = cls(interactions_df, pid_map, track_map)
        if model_exists():
            print(f"[ALS] Modelo encontrado — carregando de '{MODEL_PATH}'...")
            self.load()
        else:
            print("[ALS] Nenhum modelo salvo — treinando do zero...")
            self.train()
            self.save()
            print(f"[ALS] Modelo salvo em '{MODEL_PATH}'")
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
        from implicit.als import AlternatingLeastSquares

        # A confiança é aplicada escalando a matriz por α ANTES do fit; assim
        # a confiança efetiva vira Cui = 1 + α·r independentemente de a versão
        # do implicit expor (ou não) o parâmetro `alpha` no construtor.
        train_matrix = (self._build_user_items() * ALS_ALPHA).tocsr()

        model = AlternatingLeastSquares(
            factors=ALS_FACTORS,
            regularization=ALS_REG,
            iterations=ALS_ITERATIONS,
            random_state=SEED,
        )
        print(f"[ALS] Treinando (factors={ALS_FACTORS}, alpha={ALS_ALPHA}, "
              f"reg={ALS_REG}, iters={ALS_ITERATIONS})...")
        model.fit(train_matrix)

        self.model = model
        self._prepare_inference()

    # --------------------------------------------------------------------- #
    # PERSISTÊNCIA
    # --------------------------------------------------------------------- #
    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        self.model.save(MODEL_PATH)

    def load(self):
        # Nas versões recentes do implicit, implicit.als.AlternatingLeastSquares
        # é uma FUNÇÃO-fábrica (escolhe implementação CPU ou GPU) e não tem
        # .load() — a classe concreta com .load vive em implicit.cpu.als. Nas
        # versões antigas, implicit.als.AlternatingLeastSquares é a classe.
        try:
            from implicit.cpu.als import AlternatingLeastSquares
        except ImportError:
            from implicit.als import AlternatingLeastSquares
        self.model = AlternatingLeastSquares.load(MODEL_PATH)
        self._prepare_inference()

    def _prepare_inference(self):
        """Fatores das músicas e QᵀQ pré-computado, usados no fold-in."""
        self.item_factors = np.asarray(self.model.item_factors, dtype=np.float64)
        self._QtQ = self.item_factors.T @ self.item_factors
        self._eye = np.eye(self.item_factors.shape[1], dtype=np.float64)

    # --------------------------------------------------------------------- #
    # INFERÊNCIA
    # --------------------------------------------------------------------- #
    def recommend(self, pid_encoded, seed_idxs, exclude_idxs, top_k=500):
        """
        Top-k músicas via fold-in a partir de `seed_idxs` (músicas conhecidas
        da playlist), excluindo `exclude_idxs`. `pid_encoded` é IGNORADO — a
        interface é a mesma nos três modelos colaborativos; aqui a playlist é
        representada apenas pelas músicas seed (vetor latente em forma fechada).
        """
        Q = self.item_factors
        seed_idxs = np.asarray(seed_idxs, dtype=np.int64)
        if len(seed_idxs) == 0:
            scores = np.zeros(Q.shape[0], dtype=np.float64)
        else:
            Qr = Q[seed_idxs]                                   # (n, f)
            A  = self._QtQ + ALS_ALPHA * (Qr.T @ Qr) + ALS_REG * self._eye
            b  = (1.0 + ALS_ALPHA) * Qr.sum(axis=0)             # (f,)
            u  = np.linalg.solve(A, b)                          # (f,)
            scores = Q @ u                                      # (num_tracks,)

        exclude_idxs = np.asarray(exclude_idxs, dtype=np.int64)
        if len(exclude_idxs) > 0:
            scores[exclude_idxs] = -np.inf

        k = min(top_k, scores.shape[0])
        # argpartition: separa os k maiores sem ordenar o vetor inteiro
        part = np.argpartition(scores, -k)[-k:]
        return part[np.argsort(scores[part])[::-1]].tolist()


# =============================================================================
# EXECUÇÃO DIRETA — apenas TREINA (avaliação: compare_models.py)
# =============================================================================

if __name__ == "__main__":
    data = load_or_process_interactions()
    ALSRecommender.load_or_train(*data)
    print("[ALS] Concluído. Para avaliar/comparar: python compare_models.py")
