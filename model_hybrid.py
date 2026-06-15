# =============================================================================
# model_hybrid.py — Recomendador híbrido por fusão de scores (late fusion)
#
# Combina um modelo colaborativo (NeuMF, ALS ou Item-kNN) com o Content-Based,
# sem retreinar: compõe dois modelos já salvos em saved_models/. Expõe o mesmo
# recommend(pid, seed_idxs, exclude_idxs, top_k) dos demais modelos.
#
# Fusão (weighted hybrid):
#   1. Obtém de cada modelo o vetor de scores sobre o catálogo (score()).
#   2. Exclui as faixas já conhecidas (seeds) antes de normalizar, para o
#      min-max cobrir só as candidatas.
#   3. Normaliza cada vetor para [0,1] por min-max (as escalas diferem: NeuMF
#      é sigmoide, ALS é produto escalar, kNN é soma de similaridades, content
#      é soma ponderada).
#   4. Combina por média ponderada com peso ALPHA no colaborativo:
#          score = alpha * colab_norm + (1-alpha) * content_norm
#      apenas sobre os modelos presentes na faixa (peso renormalizado por
#      faixa): faixa sem cobertura do content usa só o colaborativo. Modelo com
#      score constante (sem poder de discriminação) também se abstém.
#
# COLAB_DEFAULT/ALPHA definem o par e o peso; HybridRecommender.build(data,
# colab_key=..., alpha=...) compõe um par específico.
#
# USO:
#   Compor:    python model_hybrid.py
#   Comparar:  python compare_models.py
# =============================================================================

import os
# BLAS em 1 thread ANTES de importar numpy — consistente com os demais modelos.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib
import numpy as np

from data_processing import load_or_process_interactions

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

MODELS_DIR = "saved_models"

ALPHA         = 0.5        # peso do colaborativo (1-ALPHA vai para o content)
COLAB_DEFAULT = "neumf"    # parceiro colaborativo do par default

# Parceiros colaborativos disponíveis: chave -> (módulo, classe, arquivo salvo).
COLAB_REGISTRY = {
    "neumf":   ("model_collaborative_neumf",   "NeuMFRecommender",
                os.path.join(MODELS_DIR, "collaborative_neumf.keras")),
    "als":     ("model_collaborative_als",     "ALSRecommender",
                os.path.join(MODELS_DIR, "collaborative_als.npz")),
    "itemknn": ("model_collaborative_itemknn", "ItemKNNRecommender",
                os.path.join(MODELS_DIR, "collaborative_itemknn.npz")),
}
CONTENT_PATH = os.path.join(MODELS_DIR, "content_based.joblib")


def required_paths(colab_key: str = COLAB_DEFAULT):
    """Arquivos que precisam existir para o híbrido estar disponível."""
    return [COLAB_REGISTRY[colab_key][2], CONTENT_PATH]


# =============================================================================
# HELPER — normalização min-max com abstenção
# =============================================================================

def _minmax(scores: np.ndarray):
    """
    Normaliza as entradas FINITAS de `scores` para [0,1]. Retorna (norm, present):
    `present` marca onde havia sinal (entrada finita E o modelo discrimina).
    Entrada -inf (modelo se abstém) vira norm=0 e present=False. Score constante
    (hi<=lo, sem poder de ranquear) também vira abstenção total.
    """
    present = np.isfinite(scores)
    norm = np.zeros_like(scores, dtype=np.float64)
    if present.any():
        v = scores[present]
        lo, hi = v.min(), v.max()
        if hi <= lo:
            present = np.zeros_like(present)   # não discrimina → abstém
        else:
            norm[present] = (v - lo) / (hi - lo)
    return norm, present


# =============================================================================
# MODELO HÍBRIDO
# =============================================================================

class HybridRecommender:
    """Fusão de scores (late fusion) entre um colaborativo e o content-based."""

    def __init__(self, colab, content, alpha: float = ALPHA, nome: str = "Híbrido"):
        self.colab      = colab
        self.content    = content
        self.alpha      = float(alpha)
        self.nome       = nome
        self.num_tracks = colab.num_tracks

    # --------------------------------------------------------------------- #
    # FACTORY — compõe dois modelos (cada um se carrega/treina sozinho)
    # --------------------------------------------------------------------- #
    @classmethod
    def build(cls, data, colab_key: str = COLAB_DEFAULT, alpha: float = ALPHA):
        from model_content import ContentRecommender
        mod_name, cls_name, _ = COLAB_REGISTRY[colab_key]
        ColabCls = getattr(importlib.import_module(mod_name), cls_name)

        print(f"[Híbrido] Compondo {cls_name} + Content "
              f"(alpha={alpha}: {alpha:.0%} colab / {1 - alpha:.0%} content)...")
        colab   = ColabCls.load_or_train(*data)
        content = ContentRecommender.load_or_train(*data)
        return cls(colab, content, alpha, nome=f"Híbrido {colab_key}+content")

    @classmethod
    def load_or_train(cls, *data):
        """Mesma assinatura dos demais modelos; compõe o par default."""
        return cls.build(data, COLAB_DEFAULT, ALPHA)

    # --------------------------------------------------------------------- #
    # INFERÊNCIA
    # --------------------------------------------------------------------- #
    def recommend(self, pid_encoded, seed_idxs, exclude_idxs, top_k=500):
        s_c = np.asarray(self.colab.score(pid_encoded, seed_idxs),   dtype=np.float64)
        s_b = np.asarray(self.content.score(pid_encoded, seed_idxs), dtype=np.float64)

        # Exclui as conhecidas ANTES de normalizar (escala = só candidatas).
        exclude_idxs = np.asarray(exclude_idxs, dtype=np.int64)
        if len(exclude_idxs) > 0:
            s_c[exclude_idxs] = -np.inf
            s_b[exclude_idxs] = -np.inf

        nc, pc = _minmax(s_c)
        nb, pb = _minmax(s_b)

        # Média ponderada SÓ sobre os modelos presentes (peso renormalizado por
        # faixa): faixa sem content usa só o colaborativo, sem ser penalizada.
        wc, wb = self.alpha, 1.0 - self.alpha
        num = wc * nc * pc + wb * nb * pb
        den = wc * pc + wb * pb

        scores = np.full(self.num_tracks, -np.inf, dtype=np.float64)
        ok = den > 0
        scores[ok] = num[ok] / den[ok]

        k = min(top_k, scores.shape[0])
        part = np.argpartition(scores, -k)[-k:]
        return part[np.argsort(scores[part])[::-1]].tolist()


# =============================================================================
# EXECUÇÃO DIRETA — só garante que os DOIS componentes estejam treinados.
# =============================================================================

if __name__ == "__main__":
    data = load_or_process_interactions()
    rec = HybridRecommender.load_or_train(*data)
    print(f"[Híbrido] Pronto: {rec.nome}. "
          f"Para avaliar/comparar: python compare_models.py")
