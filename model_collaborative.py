# =============================================================================
# model_collaborative.py — Filtragem Colaborativa pura (Neural Collaborative
# Filtering / NeuMF)
#
# Este modelo é EXCLUSIVAMENTE colaborativo: aprende a relação playlist↔música
# a partir das co-ocorrências (quais músicas aparecem juntas nas mesmas
# playlists), sem usar nenhum metadado textual (nome/artista). Serve como
# baseline para comparação com o content-based e o híbrido.
#
# CARACTERÍSTICAS:
#   1. Arquitetura NeuMF: branch GMF (linear) + branch MLP (não-linear)
#   2. BatchNormalization + Dropout para regularização
#   3. Negative sampling seguro (negativos nunca são músicas da própria playlist)
#   4. Learning rate scheduling (ReduceLROnPlateau) + Early stopping
#   5. Reprodutibilidade controlada por SEED
#
# REQUISITOS: veja requirements.txt
#
# USO:
#   Execução direta:   python model_collaborative.py
#   Como biblioteca:
#       from model_collaborative import CollaborativeRecommender
#       rec = CollaborativeRecommender.load_or_train(...)
#       rec.evaluate()
# =============================================================================

import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from keras.layers import (Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Multiply)
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

from data_processing import load_dataset, build_interactions

# =============================================================================
# CAMINHOS DE PERSISTÊNCIA
# =============================================================================

MODELS_DIR = "saved_models"
MODEL_PATH = os.path.join(MODELS_DIR, "collaborative_model.keras")

# =============================================================================
# VARIÁVEIS DE TREINAMENTO
# =============================================================================

SEED          = 42      # Semente única para reprodutibilidade (treino e avaliação)
EMBEDDING_DIM = 64      # Dimensão dos embeddings
MLP_LAYERS    = [256, 128, 64]  # Camadas da branch MLP
DROPOUT_RATE  = 0.3     # Taxa de dropout para regularização
L2_REG        = 1e-6    # Regularização L2 nos embeddings
EPOCHS        = 25      # Máximo de épocas (early stopping pode parar antes)
BATCH_SIZE    = 4096
TEST_SIZE     = 0.1
NEG_RATIO     = 1       # Quantas amostras negativas por positiva

# =============================================================================
# CONSTRUÇÃO DO MODELO — Neural Matrix Factorization (NeuMF)
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
# PERSISTÊNCIA — SALVAR E CARREGAR
# =============================================================================

def save_model(model: Model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo colaborativo salvo em '{MODEL_PATH}'")


def load_model_from_disk() -> Model:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\n"
            "Execute train() e save_model() primeiro."
        )
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo colaborativo carregado de '{MODEL_PATH}'")
    return model


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)

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
    print(f"Gerando amostras de treinamento (negative sampling seguro, "
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

    save_model(model)
    return model

# =============================================================================
# INFERÊNCIA — NeuMF puro
# =============================================================================

def recommend(
    model: Model,
    pid_encoded: int,
    num_tracks: int,
    exclude_idxs: list,
    top_k: int = 200,
) -> list:
    """
    Ranqueia todas as tracks pelo score NeuMF para um dado pid_encoded.
    Exclui os índices em exclude_idxs (tracks que já estão na playlist).
    Retorna os top_k índices em ordem decrescente de score.
    """
    all_track_idxs = np.arange(num_tracks)

    scores = model.predict(
        [np.full(num_tracks, pid_encoded), all_track_idxs],
        batch_size=4096, verbose=0
    ).flatten()

    scores[exclude_idxs] = -1
    top_indices = scores.argsort()[-top_k:][::-1]
    return top_indices.tolist()

# =============================================================================
# VARIÁVEIS DE AVALIAÇÃO
# =============================================================================

NUM_PLAYLISTS_TEST = 5
PCT_REMOVED        = 0.20
TOP_K_REC          = 500

# =============================================================================
# CLASSE — CollaborativeRecommender
# Interface orientada a objetos sobre as funções acima: encapsula o modelo
# treinado e expõe load_or_train() e evaluate() de forma autocontida.
# =============================================================================

class CollaborativeRecommender:
    """Wrapper sobre o modelo NeuMF puro — baseline para comparações."""

    def __init__(self, model, interactions_df, pid_map, reverse_track_map, uri_to_name):
        self.model             = model
        self.interactions_df   = interactions_df
        self.pid_map           = pid_map
        self.reverse_track_map = reverse_track_map
        self.uri_to_name       = uri_to_name
        self.num_tracks        = len(reverse_track_map)

    @classmethod
    def load_or_train(cls, interactions_df, pid_map, track_map, reverse_track_map, uri_to_name):
        """Factory: carrega o modelo salvo ou treina do zero."""
        num_playlists = len(pid_map)
        num_tracks    = len(track_map)
        if model_exists():
            print("Modelo colaborativo encontrado — carregando do disco...")
            model = load_model_from_disk()
        else:
            print("Nenhum modelo salvo — treinando do zero...")
            model = train(interactions_df, num_playlists, num_tracks)
        return cls(model, interactions_df, pid_map, reverse_track_map, uri_to_name)

    def evaluate(
        self,
        num_playlists_test: int = NUM_PLAYLISTS_TEST,
        pct_removed: float      = PCT_REMOVED,
        top_k: int              = TOP_K_REC,
        seed: int               = SEED,
    ) -> float:
        """
        Avalia o modelo nos primeiros `num_playlists_test` PIDs disponíveis.
        Para cada playlist remove `pct_removed`% das músicas e mede quantas
        o modelo recupera no Top K (Recall@K).

        O `seed` fixa quais músicas são removidas, garantindo que execuções
        diferentes (e modelos diferentes) sejam avaliadas sobre o MESMO
        conjunto removido — comparação justa entre baselines.

        Retorna o Recall@K geral (em %).
        """
        rng           = np.random.default_rng(seed)
        test_pids     = self.interactions_df['pid'].unique()[:num_playlists_test]
        total_hits    = 0
        total_removed = 0

        for pid in test_pids:
            actual_tracks = self.interactions_df[
                self.interactions_df['pid'] == pid
            ]['track_encoded'].values

            n_rem     = max(1, int(len(actual_tracks) * pct_removed))
            removed   = rng.choice(actual_tracks, n_rem, replace=False)
            remaining = [t for t in actual_tracks if t not in removed]

            print(f"\n{'='*60}")
            print(f"Playlist PID: {pid} | Total: {len(actual_tracks)} músicas")
            print(f"Removidas: {len(removed)} | Restantes: {len(remaining)}")
            print("Músicas removidas:")
            for r_idx in removed:
                print(f"  - {self.uri_to_name.get(self.reverse_track_map[r_idx], '?')}")

            p_idx = self.pid_map[pid]
            top_indices = recommend(
                model=self.model,
                pid_encoded=p_idx,
                num_tracks=self.num_tracks,
                exclude_idxs=remaining,
                top_k=top_k,
            )

            removed_set = set(removed.tolist())
            hits        = [idx for idx in top_indices if idx in removed_set]
            recall      = len(hits) / len(removed) * 100
            total_hits    += len(hits)
            total_removed += len(removed)

            print(f"\nRecuperadas no Top {top_k}: {len(hits)}/{len(removed)} "
                  f"→ Recall@{top_k} = {recall:.1f}%")
            if hits:
                print("Músicas recuperadas:")
                for idx in hits:
                    print(f"  {self.uri_to_name.get(self.reverse_track_map[idx], '?')}")
            else:
                print("Nenhuma música recuperada.")

        overall_recall = total_hits / total_removed * 100 if total_removed > 0 else 0
        print(f"\n{'='*60}")
        print(f"RECALL GERAL @{top_k}: {total_hits}/{total_removed} = {overall_recall:.1f}%")
        return overall_recall


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    df = load_dataset()
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)

    rec = CollaborativeRecommender.load_or_train(
        interactions_df, pid_map, track_map, reverse_track_map, uri_to_name
    )
    rec.evaluate()