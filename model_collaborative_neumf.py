# =============================================================================
# model_collaborative_neumf.py — Filtragem Colaborativa pura via rede neural
# (Neural Collaborative Filtering / NeuMF)
#
# (antes chamado model_collaborative.py — renomeado para deixar explícito QUAL
# modelo colaborativo este arquivo implementa, como o _als e o _itemknn)
#
# ARQUIVO AUTOCONTIDO: este arquivo só TREINA o modelo e expõe a inferência
# (recommend). A lógica de teste/avaliação — escolha das playlists, remoção
# de músicas e métricas — vive em compare_collaborative.py, que usa o MESMO
# conjunto de teste para todos os modelos (comparação justa).
#
# Modelo EXCLUSIVAMENTE colaborativo: aprende a relação playlist↔música a
# partir das co-ocorrências (quais músicas aparecem juntas nas mesmas
# playlists), sem usar nenhum metadado textual (nome/artista).
#
# CARACTERÍSTICAS:
#   1. Arquitetura NeuMF: branch GMF (linear) + branch MLP (não-linear)
#   2. BatchNormalization + Dropout para regularização
#   3. Negative sampling seguro (negativos nunca são músicas da própria playlist)
#   4. Learning rate scheduling (ReduceLROnPlateau) + Early stopping
#   5. Reprodutibilidade controlada por SEED
#
# USO:
#   Treinar:   python model_collaborative_neumf.py   (ou via main.py)
#   Comparar:  python compare_collaborative.py
# =============================================================================

import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from keras.layers import (Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Multiply)
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

from data_processing import load_or_process_interactions

# =============================================================================
# PERSISTÊNCIA E HIPERPARÂMETROS
# =============================================================================

MODELS_DIR = "saved_models"
MODEL_PATH = os.path.join(MODELS_DIR, "collaborative_neumf.keras")

SEED          = 42      # Semente única para reprodutibilidade do treino
EMBEDDING_DIM = 64      # Dimensão dos embeddings
MLP_LAYERS    = [256, 128, 64]  # Camadas da branch MLP
DROPOUT_RATE  = 0.3     # Taxa de dropout para regularização
L2_REG        = 1e-6    # Regularização L2 nos embeddings
EPOCHS        = 25      # Máximo de épocas (early stopping pode parar antes)
BATCH_SIZE    = 4096
TEST_SIZE     = 0.1     # Fração para validação interna do treino (val_loss)
NEG_RATIO     = 1       # Quantas amostras negativas por positiva


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

# =============================================================================
# CONSTRUÇÃO DA REDE — Neural Matrix Factorization (NeuMF)
# =============================================================================
#
# NeuMF combina duas branches:
#   • GMF (Generalized Matrix Factorization): produto elemento a elemento
#     dos embeddings → captura interações lineares simples
#   • MLP (Multi-Layer Perceptron): concatenação + camadas densas
#     → captura padrões não-lineares complexos
#
# As duas saídas são concatenadas e passam por uma camada final sigmoid.
# =============================================================================

def build_model(num_playlists: int, num_tracks: int,
                embedding_dim: int = EMBEDDING_DIM) -> Model:
    """Constrói e compila o modelo NeuMF."""

    # --- Inputs ---
    p_input = Input(shape=(1,), name="playlist_input")
    t_input = Input(shape=(1,), name="track_input")

    # ------------------------------------------------------------------ #
    # Branch 1 — GMF (Generalized Matrix Factorization)                   #
    # Captura relações lineares via produto elemento a elemento            #
    # ------------------------------------------------------------------ #
    p_emb_gmf = Embedding(num_playlists, embedding_dim,
                          embeddings_regularizer=l2(L2_REG),
                          name="playlist_emb_gmf")(p_input)
    t_emb_gmf = Embedding(num_tracks, embedding_dim,
                          embeddings_regularizer=l2(L2_REG),
                          name="track_emb_gmf")(t_input)
    gmf_out = Multiply()([Flatten()(p_emb_gmf), Flatten()(t_emb_gmf)])

    # ------------------------------------------------------------------ #
    # Branch 2 — MLP (Multi-Layer Perceptron)                             #
    # Captura padrões não-lineares com camadas densas empilhadas           #
    # ------------------------------------------------------------------ #
    p_emb_mlp = Embedding(num_playlists, embedding_dim,
                          embeddings_regularizer=l2(L2_REG),
                          name="playlist_emb_mlp")(p_input)
    t_emb_mlp = Embedding(num_tracks, embedding_dim,
                          embeddings_regularizer=l2(L2_REG),
                          name="track_emb_mlp")(t_input)
    mlp_out = Concatenate()([Flatten()(p_emb_mlp), Flatten()(t_emb_mlp)])

    for units in MLP_LAYERS:
        mlp_out = Dense(units, activation='relu',
                        kernel_regularizer=l2(L2_REG))(mlp_out)
        mlp_out = BatchNormalization()(mlp_out)
        mlp_out = Dropout(DROPOUT_RATE)(mlp_out)

    # ------------------------------------------------------------------ #
    # Fusão GMF + MLP → camada de saída                                   #
    # ------------------------------------------------------------------ #
    combined = Concatenate()([gmf_out, mlp_out])
    out = Dense(1, activation='sigmoid', name="output")(combined)

    model = Model(inputs=[p_input, t_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# =============================================================================
# NEGATIVE SAMPLING SEGURO
# =============================================================================

def build_training_data(interactions_df, num_tracks: int,
                        neg_ratio: int = NEG_RATIO, seed: int = SEED):
    """
    Gera pares positivos e negativos garantindo que os negativos
    não sejam músicas que a playlist realmente contém.

    Para cada par positivo (playlist, música), sorteia `neg_ratio` músicas
    negativas que estão FORA da playlist — eliminando falsos negativos que
    confundem o treinamento.

    Otimização: o sorteio é vetorizado (numpy) e apenas as colisões são
    re-sorteadas, evitando o laço Python por linha da versão anterior.
    """
    print(f"[NeuMF] Gerando amostras de treino (negative sampling seguro, "
          f"neg_ratio={neg_ratio})...")

    rng = np.random.default_rng(seed)

    pos_pids   = interactions_df['pid_encoded'].values
    pos_tracks = interactions_df['track_encoded'].values

    # Índice: pid_encoded → set de track_encoded que a playlist contém
    pid_to_tracks = (
        interactions_df
        .groupby('pid_encoded')['track_encoded']
        .apply(set)
        .to_dict()
    )

    # Cada positivo gera `neg_ratio` negativos → replica os pids.
    # int32 nos índices e float32 nos rótulos cortam o uso de RAM pela metade
    # em relação aos defaults int64/float64 do numpy.
    neg_pids   = np.repeat(pos_pids, neg_ratio)
    neg_tracks = rng.integers(0, num_tracks, size=len(neg_pids), dtype=np.int32)

    # Re-sorteia apenas as colisões (negativo que a playlist de fato contém)
    collision_idx = np.array([
        i for i, (p, t) in enumerate(zip(neg_pids, neg_tracks))
        if t in pid_to_tracks[p]
    ])
    while len(collision_idx) > 0:
        neg_tracks[collision_idx] = rng.integers(
            0, num_tracks, size=len(collision_idx), dtype=np.int32)
        collision_idx = collision_idx[[
            neg_tracks[i] in pid_to_tracks[neg_pids[i]] for i in collision_idx
        ]]

    X_pids   = np.concatenate([pos_pids,   neg_pids]).astype(np.int32)
    X_tracks = np.concatenate([pos_tracks, neg_tracks]).astype(np.int32)
    y        = np.concatenate([
        np.ones(len(pos_pids), dtype=np.float32),
        np.zeros(len(neg_pids), dtype=np.float32),
    ])

    # Embaralha tudo junto
    idx = rng.permutation(len(y))
    return X_pids[idx], X_tracks[idx], y[idx]

# =============================================================================
# TREINAMENTO
# =============================================================================

def train(interactions_df, num_playlists: int, num_tracks: int) -> Model:
    """
    Treina o modelo NeuMF com callbacks de LR scheduling e early stopping.
    """
    # Semeia numpy, random e TensorFlow de uma vez — torna a inicialização
    # dos pesos e o dropout reprodutíveis entre execuções.
    keras.utils.set_random_seed(SEED)

    X_pids, X_tracks, y = build_training_data(interactions_df, num_tracks)

    X_p_train, X_p_val, X_t_train, X_t_val, y_train, y_val = train_test_split(
        X_pids, X_tracks, y, test_size=TEST_SIZE, random_state=SEED
    )

    model = build_model(num_playlists, num_tracks, EMBEDDING_DIM)
    model.summary()

    callbacks = [
        # Reduz o learning rate quando a val_loss para de melhorar
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-5, verbose=1),
        # Para o treino cedo se não houver melhora por 4 épocas seguidas
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

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[NeuMF] Modelo salvo em '{MODEL_PATH}'")
    return model

# =============================================================================
# CLASSE — NeuMFRecommender
# Mesma interface dos demais modelos colaborativos (ALSRecommender,
# ItemKNNRecommender): load_or_train() + recommend().
# =============================================================================

class NeuMFRecommender:
    """Filtragem colaborativa via rede neural NeuMF (GMF + MLP)."""

    def __init__(self, model: Model, num_tracks: int):
        self.model      = model
        self.num_tracks = num_tracks

    @classmethod
    def load_or_train(cls, interactions_df, pid_map, track_map,
                      reverse_track_map=None, uri_to_name=None):
        """
        Assinatura idêntica nos três modelos colaborativos (drop-in no main.py
        e no compare_collaborative.py). reverse_track_map e uri_to_name não
        são usados aqui — nomes de músicas só importam no relatório do compare.
        """
        num_playlists = len(pid_map)
        num_tracks    = len(track_map)
        if model_exists():
            print(f"[NeuMF] Modelo encontrado — carregando de '{MODEL_PATH}'...")
            model = keras.models.load_model(MODEL_PATH)
        else:
            print("[NeuMF] Nenhum modelo salvo — treinando do zero...")
            model = train(interactions_df, num_playlists, num_tracks)
        return cls(model, num_tracks)

    # --------------------------------------------------------------------- #
    # INFERÊNCIA
    # --------------------------------------------------------------------- #
    def recommend(self, pid_encoded, seed_idxs, exclude_idxs, top_k=500):
        """
        Top-k músicas pelo score NeuMF para `pid_encoded`, excluindo
        `exclude_idxs`. `seed_idxs` é IGNORADO — o NeuMF representa a playlist
        pelo embedding aprendido no treino (id da playlist), não pelas músicas
        seed; o parâmetro existe para a interface ser a mesma nos três modelos.
        """
        all_track_idxs = np.arange(self.num_tracks)
        scores = self.model.predict(
            [np.full(self.num_tracks, pid_encoded), all_track_idxs],
            batch_size=4096, verbose=0
        ).flatten().astype(np.float64)

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
    NeuMFRecommender.load_or_train(*data)
    print("[NeuMF] Concluído. Para avaliar/comparar: python compare_collaborative.py")
