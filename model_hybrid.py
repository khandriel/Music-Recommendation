# =============================================================================
# model_hybrid.py -- Modelo Hibrido (Colaborativo + Content-Based)
#
# Este modelo NAO reescreve nenhum dos dois modelos originais. Ele os JUNTA:
#
#   - BASE TREINAVEL: o NeuMF do model_collaborative.py (importado intacto).
#     A arquitetura (build_model), o negative sampling seguro
#     (build_training_data), a inferencia (recommend) e a validacao
#     (CollaborativeRecommender.evaluate) sao reaproveitados sem alteracao.
#
#   - AJUDA AO TREINO: o content-based entra ENRIQUECENDO os dados de treino.
#     Para cada playlist, ele sugere faixas content-similares as que ja estao
#     na playlist; essas faixas entram como POSITIVOS SUAVES (soft labels) no
#     treino do NeuMF. Assim o conhecimento de conteudo (audio, ano, genero...)
#     e injetado no colaborativo sem mudar nenhuma das duas arquiteturas.
#
# A ponte entre os dois mundos e a URI da faixa:
#     colaborativo:  track_encoded  <->  reverse_track_map / track_map  <->  URI
#     content-based: linha do df    <->  coluna 'uri'                   <->  URI
#
# VALIDACAO: herdada de CollaborativeRecommender.evaluate (Recall@K), conforme
# pedido. HybridRecommender e subclasse de CollaborativeRecommender e so
# sobrescreve o treino; evaluate() continua identico.
#
# USO:
#   python model_hybrid.py
#   ou:
#       from model_hybrid import HybridRecommender, ContentScorer
#       scorer = ContentScorer.from_csv("audio_features.csv")
#       rec = HybridRecommender.load_or_train(..., content_scorer=scorer)
#       rec.evaluate()
# =============================================================================

import os
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# -----------------------------------------------------------------------------
# Reaproveitamento direto do modelo colaborativo (NADA aqui e reimplementado).
# -----------------------------------------------------------------------------
from data_processing import load_dataset, build_interactions
from model_collaborative import (
    build_model,            # arquitetura NeuMF, intacta
    build_training_data,    # positivos + negative sampling seguro, intacto
    recommend,              # inferencia NeuMF, usada pela validacao herdada
    CollaborativeRecommender,
    SEED, EMBEDDING_DIM, EPOCHS, BATCH_SIZE, TEST_SIZE,
)

# =============================================================================
# CAMINHOS DE PERSISTENCIA (separados do colaborativo para nao sobrescrever)
# =============================================================================

MODELS_DIR        = "saved_models"
HYBRID_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_model.keras")

# CSV de conteudo (uma linha por faixa, com coluna 'uri' + features de audio).
# Ajuste o caminho conforme o seu ambiente.
CONTENT_CSV_PATH = "audio_features.csv"

# =============================================================================
# PARAMETROS DA PONTE CONTENT-BASED -> TREINO
# =============================================================================

TOP_N_SIMILAR     = 10      # faixas content-similares buscadas por faixa-semente
MAX_PLAYLISTS_AUG = None    # limite de playlists usadas no augmentation (None = todas)
MAX_SEED_TRACKS   = 20      # faixas-semente por playlist (controla custo)
SOFT_LABEL        = 0.5     # rotulo dos positivos sugeridos pelo conteudo (0 < x <= 1)

# Mesmas features de audio do contentbasedv3.py.
AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'tempo',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'key', 'mode',
]
SIGMA_ANO = 15  # desvio do kernel gaussiano de ano (igual ao content-based)

# Pesos default do scorer. Espelham a logica do contentbasedv3 (soma ponderada
# de scores independentes), priorizando os sinais vetorizaveis e rapidos para
# o augmentation. 'genero' fica disponivel mas desligado por padrao (Jaccard em
# laco e custoso quando rodado para milhares de faixas-semente).
CONTENT_WEIGHTS_DEFAULT = {'audio': 0.80, 'ano': 0.20}


# =============================================================================
# ContentScorer -- adaptador do content-based, por URI
#
# Encapsula a MESMA formula do contentbasedv3 (soma ponderada de similaridades
# independentes), exposta de um jeito que o hibrido consegue consultar:
# "dada uma URI, quais URIs sao mais content-similares?". O modelo content-based
# em si nao e alterado; isto e apenas um adaptador reaproveitavel.
# =============================================================================

class ContentScorer:
    def __init__(self, content_df: pd.DataFrame, weights: dict = None,
                 sigma_ano: float = SIGMA_ANO):
        if 'uri' not in content_df.columns:
            raise ValueError("O CSV de conteudo precisa de uma coluna 'uri'.")

        self.df        = content_df.reset_index(drop=True)
        self.weights   = dict(weights or CONTENT_WEIGHTS_DEFAULT)
        self.sigma_ano = sigma_ano

        # URI -> indice de linha (para localizar a faixa-alvo rapidamente)
        self.uri_to_idx = {u: i for i, u in enumerate(self.df['uri'].values)}

        # --- Matriz de audio padronizada + normalizada em L2 (igual content) ---
        audio_cols = [c for c in AUDIO_FEATURES if c in self.df.columns]
        if audio_cols:
            std = StandardScaler().fit_transform(
                self.df[audio_cols].astype(np.float32)
            ).astype(np.float32)
            normas = np.linalg.norm(std, axis=1, keepdims=True)
            normas[normas == 0] = 1.0
            self.mat_audio = std / normas
        else:
            self.mat_audio = None
            print("ContentScorer: nenhuma feature de audio encontrada no CSV.")

        # --- Ano (derivado do ISRC, mesma logica do contentbasedv3) -----------
        if 'ano' in self.df.columns:
            self.ano = self.df['ano'].astype(np.float32).values
        elif 'isrc' in self.df.columns:
            anos = self.df['isrc'].apply(self._isrc_para_ano)
            self.ano = anos.fillna(anos.median()).astype(np.float32).values
        else:
            self.ano = None

        # --- Genero (lista por faixa; opcional, usado so se peso > 0) ----------
        if 'genero_lista' in self.df.columns:
            self.genero = self.df['genero_lista'].apply(
                lambda x: set(x) if isinstance(x, (list, set, tuple)) else set()
            ).values
        else:
            self.genero = None

        # Zera pesos de sinais ausentes para evitar contribuicoes vazias.
        if self.mat_audio is None:
            self.weights.pop('audio', None)
        if self.ano is None:
            self.weights.pop('ano', None)
        if self.genero is None:
            self.weights.pop('genero', None)

        total = sum(v for v in self.weights.values() if v > 0)
        if total <= 0:
            raise ValueError("ContentScorer sem sinais utilizaveis (pesos zerados).")
        self.weights = {k: v / total for k, v in self.weights.items() if v > 0}

        print(f"ContentScorer pronto: {len(self.df):,} faixas | "
              f"pesos normalizados = {self.weights}")

    # --- Construtores ------------------------------------------------------- #

    @classmethod
    def from_csv(cls, path: str, weights: dict = None) -> "ContentScorer":
        """Carrega o CSV de conteudo (uma faixa por linha, coluna 'uri')."""
        df = pd.read_csv(path, dtype={'uri': str, 'isrc': str,
                                      'nome': str, 'artista': str})
        cols_audio = [c for c in AUDIO_FEATURES if c in df.columns]
        if cols_audio:
            df = df.dropna(subset=cols_audio).reset_index(drop=True)
        print(f"CSV de conteudo carregado: {len(df):,} faixas de '{path}'")
        return cls(df, weights=weights)

    # --- Utilitarios internos ---------------------------------------------- #

    @staticmethod
    def _isrc_para_ano(isrc):
        """Mesma extracao de ano a partir do ISRC usada no contentbasedv3."""
        try:
            s = str(isrc).strip().replace('-', '')
            if len(s) < 7:
                return np.nan
            yy = int(s[5:7])
            return 2000 + yy if yy <= 25 else 1900 + yy
        except Exception:
            return np.nan

    def has_uri(self, uri: str) -> bool:
        return uri in self.uri_to_idx

    # --- Score principal ---------------------------------------------------- #

    def _score_vector(self, idx_alvo: int) -> np.ndarray:
        """Score content-based da faixa-alvo contra toda a base, em [0, 1]."""
        n = len(self.df)
        score = np.zeros(n, dtype=np.float32)

        if 'audio' in self.weights and self.mat_audio is not None:
            sim = (self.mat_audio[idx_alvo:idx_alvo + 1] @ self.mat_audio.T).flatten()
            sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
            score += self.weights['audio'] * sim

        if 'ano' in self.weights and self.ano is not None:
            diff = (self.ano - self.ano[idx_alvo]) / self.sigma_ano
            score += self.weights['ano'] * np.exp(-0.5 * diff ** 2).astype(np.float32)

        if 'genero' in self.weights and self.genero is not None:
            alvo = self.genero[idx_alvo]
            if alvo:
                gscore = np.zeros(n, dtype=np.float32)
                for i, cand in enumerate(self.genero):
                    if cand:
                        uni = len(alvo | cand)
                        if uni:
                            gscore[i] = len(alvo & cand) / uni
                score += self.weights['genero'] * gscore

        return score

    def top_similar(self, uri: str, top_n: int = TOP_N_SIMILAR):
        """Retorna [(uri, score), ...] das top_n faixas mais content-similares."""
        idx_alvo = self.uri_to_idx.get(uri)
        if idx_alvo is None:
            return []

        score = self._score_vector(idx_alvo)
        score[idx_alvo] = -1.0

        k = min(top_n, len(score) - 1)
        if k <= 0:
            return []
        part = np.argpartition(score, -k)[-k:]
        order = part[np.argsort(score[part])[::-1]]
        uris = self.df['uri'].values
        return [(uris[i], float(score[i])) for i in order]


# =============================================================================
# AUGMENTATION -- gera positivos suaves a partir do content-based
# =============================================================================

def build_content_augmented_pairs(
    interactions_df,
    content_scorer: ContentScorer,
    track_map: dict,
    reverse_track_map: dict,
    top_n_similar: int = TOP_N_SIMILAR,
    max_playlists=MAX_PLAYLISTS_AUG,
    max_seed_tracks: int = MAX_SEED_TRACKS,
    seed: int = SEED,
):
    """
    Para cada playlist, usa o content-based para encontrar faixas similares as
    que ela ja contem e propoe essas faixas como positivos adicionais.

    Filtros:
      - so propoe faixas conhecidas pelo colaborativo (presentes em track_map)
      - nunca propoe faixa que a playlist ja contem (evita duplicar positivo real)
      - cacheia top_similar por URI (mesma faixa aparece em varias playlists)

    Retorna (aug_pids, aug_tracks): arrays int32 de pares (playlist, faixa).
    """
    print(f"Augmentation content-based (top_n={top_n_similar}, "
          f"soft_label={SOFT_LABEL})...")

    rng = np.random.default_rng(seed)

    # pid_encoded -> set de track_encoded que a playlist contem
    pid_to_tracks = (
        interactions_df
        .groupby('pid_encoded')['track_encoded']
        .apply(set)
        .to_dict()
    )

    pids = list(pid_to_tracks.keys())
    if max_playlists is not None and max_playlists < len(pids):
        pids = rng.choice(pids, size=max_playlists, replace=False).tolist()

    sims_cache: dict = {}
    aug_pids, aug_tracks = [], []

    for pid in pids:
        owned = pid_to_tracks[pid]

        seeds = list(owned)
        if max_seed_tracks is not None and len(seeds) > max_seed_tracks:
            seeds = rng.choice(seeds, size=max_seed_tracks, replace=False).tolist()

        added = set()
        for t_enc in seeds:
            uri = reverse_track_map.get(t_enc)
            if uri is None or not content_scorer.has_uri(uri):
                continue

            if uri in sims_cache:
                sims = sims_cache[uri]
            else:
                sims = content_scorer.top_similar(uri, top_n_similar)
                sims_cache[uri] = sims

            for sim_uri, _score in sims:
                enc = track_map.get(sim_uri)
                if enc is None or enc in owned or enc in added:
                    continue
                added.add(enc)
                aug_pids.append(pid)
                aug_tracks.append(enc)

    aug_pids   = np.asarray(aug_pids,   dtype=np.int32)
    aug_tracks = np.asarray(aug_tracks, dtype=np.int32)
    print(f"  -> {len(aug_pids):,} positivos suaves gerados pelo conteudo")
    return aug_pids, aug_tracks


def build_hybrid_training_data(
    interactions_df,
    num_tracks: int,
    content_scorer: ContentScorer = None,
    track_map: dict = None,
    reverse_track_map: dict = None,
    seed: int = SEED,
):
    """
    Monta o conjunto de treino do hibrido:

      1. Base colaborativa intacta: build_training_data() do model_collaborative
         (positivos reais + negative sampling seguro).
      2. Positivos suaves vindos do content-based (se houver scorer).

    Sem scorer, e identico ao treino colaborativo puro (fallback seguro).
    """
    X_pids, X_tracks, y = build_training_data(interactions_df, num_tracks)

    if content_scorer is None:
        print("Sem ContentScorer: treino identico ao colaborativo puro.")
        return X_pids, X_tracks, y

    aug_pids, aug_tracks = build_content_augmented_pairs(
        interactions_df, content_scorer, track_map, reverse_track_map, seed=seed,
    )
    if len(aug_pids) == 0:
        print("Nenhum positivo suave gerado; usando apenas a base colaborativa.")
        return X_pids, X_tracks, y

    aug_y = np.full(len(aug_pids), SOFT_LABEL, dtype=np.float32)

    X_pids   = np.concatenate([X_pids,   aug_pids]).astype(np.int32)
    X_tracks = np.concatenate([X_tracks, aug_tracks]).astype(np.int32)
    y        = np.concatenate([y, aug_y]).astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    print(f"Treino hibrido: {len(y):,} exemplos "
          f"({len(aug_pids):,} suaves do conteudo)")
    return X_pids[idx], X_tracks[idx], y[idx]


# =============================================================================
# PERSISTENCIA DO HIBRIDO
# =============================================================================

def hybrid_model_exists() -> bool:
    return os.path.exists(HYBRID_MODEL_PATH)


def save_hybrid_model(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(HYBRID_MODEL_PATH)
    print(f"Modelo hibrido salvo em '{HYBRID_MODEL_PATH}'")


def load_hybrid_model_from_disk():
    if not hybrid_model_exists():
        raise FileNotFoundError(
            f"Modelo hibrido nao encontrado: {HYBRID_MODEL_PATH}"
        )
    model = keras.models.load_model(HYBRID_MODEL_PATH)
    print(f"Modelo hibrido carregado de '{HYBRID_MODEL_PATH}'")
    return model


# =============================================================================
# TREINAMENTO DO HIBRIDO
# Mesma estrutura do train() colaborativo (build_model intacto, mesmos
# callbacks), mas alimentado pelos dados enriquecidos pelo conteudo.
# =============================================================================

def train_hybrid(
    interactions_df,
    num_playlists: int,
    num_tracks: int,
    track_map: dict = None,
    reverse_track_map: dict = None,
    content_scorer: ContentScorer = None,
):
    keras.utils.set_random_seed(SEED)

    X_pids, X_tracks, y = build_hybrid_training_data(
        interactions_df, num_tracks,
        content_scorer=content_scorer,
        track_map=track_map,
        reverse_track_map=reverse_track_map,
    )

    X_p_train, X_p_val, X_t_train, X_t_val, y_train, y_val = train_test_split(
        X_pids, X_tracks, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = build_model(num_playlists, num_tracks, EMBEDDING_DIM)
    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=4,
                      restore_best_weights=True, verbose=1),
    ]

    model.fit(
        [X_p_train, X_t_train], y_train,
        validation_data=([X_p_val, X_t_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    save_hybrid_model(model)
    return model


# =============================================================================
# CLASSE -- HybridRecommender
#
# Subclasse de CollaborativeRecommender: herda a inferencia e a VALIDACAO
# (evaluate / Recall@K) sem nenhuma alteracao, e apenas troca a logica de
# treino para usar os dados enriquecidos pelo content-based.
# =============================================================================

class HybridRecommender(CollaborativeRecommender):

    @classmethod
    def load_or_train(cls, interactions_df, pid_map, track_map,
                      reverse_track_map, uri_to_name, content_scorer=None):
        num_playlists = len(pid_map)
        num_tracks    = len(track_map)

        if hybrid_model_exists():
            print("Modelo hibrido encontrado -- carregando do disco...")
            model = load_hybrid_model_from_disk()
        else:
            print("Nenhum modelo hibrido salvo -- treinando do zero...")
            if content_scorer is None:
                print("AVISO: sem ContentScorer, o treino sera colaborativo puro.")
            model = train_hybrid(
                interactions_df, num_playlists, num_tracks,
                track_map=track_map,
                reverse_track_map=reverse_track_map,
                content_scorer=content_scorer,
            )

        # __init__ herdado de CollaborativeRecommender
        return cls(model, interactions_df, pid_map, reverse_track_map, uri_to_name)

    # evaluate() NAO e sobrescrito: usa a validacao do colaborativo.


# =============================================================================
# EXECUCAO DIRETA
# =============================================================================

if __name__ == "__main__":
    df = load_dataset()
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = \
        build_interactions(df)

    # Carrega o conteudo por URI. Se o CSV nao existir, cai no colaborativo puro.
    content_scorer = None
    if os.path.exists(CONTENT_CSV_PATH):
        content_scorer = ContentScorer.from_csv(CONTENT_CSV_PATH)
    else:
        print(f"AVISO: CSV de conteudo '{CONTENT_CSV_PATH}' nao encontrado. "
              f"Treinando hibrido em modo fallback (colaborativo puro).")

    rec = HybridRecommender.load_or_train(
        interactions_df, pid_map, track_map, reverse_track_map, uri_to_name,
        content_scorer=content_scorer,
    )
    rec.evaluate()