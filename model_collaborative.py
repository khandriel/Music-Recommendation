# =============================================================================
# model_collaborative.py — Filtragem Colaborativa (Embedding)
#
# REQUISITOS:
#   pip install numpy scikit-learn tensorflow keras
#
# USO:
#   Execute diretamente:  python model_collaborative.py
#   Ou importe as funções em outro script:
#       from model_collaborative import train, load_model_from_disk, evaluate
# =============================================================================

import numpy as np
import os
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Activation
from keras.models import Model

from data_processing import load_dataset, build_interactions

# =============================================================================
# CAMINHOS DE PERSISTÊNCIA
# =============================================================================

MODELS_DIR  = "saved_models"
MODEL_PATH  = os.path.join(MODELS_DIR, "collaborative_model.keras")

# =============================================================================
# VARIÁVEIS DE TREINAMENTO
# =============================================================================

EMBEDDING_DIM = 64
EPOCHS        = 10
BATCH_SIZE    = 4096
TEST_SIZE     = 0.2

# =============================================================================
# CONSTRUÇÃO DO MODELO
# =============================================================================

def build_model(num_playlists: int, num_tracks: int, embedding_dim: int) -> Model:
    """Constrói e compila o modelo de filtragem colaborativa por embeddings."""
    p_input = Input(shape=(1,))
    t_input = Input(shape=(1,))
    p_emb   = Embedding(num_playlists, embedding_dim)(p_input)
    t_emb   = Embedding(num_tracks,    embedding_dim)(t_input)
    dot     = Dot(axes=2)([p_emb, t_emb])
    out     = Activation('sigmoid')(Flatten()(dot))

    model = Model(inputs=[p_input, t_input], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================================================================
# PERSISTÊNCIA — SALVAR E CARREGAR
# =============================================================================

def save_model(model: Model):
    """Salva o modelo Keras treinado em saved_models/."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modelo colaborativo salvo em '{MODEL_PATH}'")


def load_model_from_disk() -> Model:
    """
    Carrega o modelo Keras salvo em disco.
    Lança FileNotFoundError se o arquivo não existir.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\n"
            "Execute train() e save_model() primeiro."
        )
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo colaborativo carregado de '{MODEL_PATH}'")
    return model


def model_exists() -> bool:
    """Retorna True se o modelo já foi salvo em disco."""
    return os.path.exists(MODEL_PATH)

# =============================================================================
# TREINAMENTO
# =============================================================================

def train(interactions_df, num_playlists: int, num_tracks: int) -> Model:
    """
    Gera pares positivos/negativos, divide em treino/validação e treina o modelo.
    Salva o modelo treinado em disco e o retorna.
    """
    pos_pids   = interactions_df['pid_encoded'].values
    pos_tracks = interactions_df['track_encoded'].values
    neg_tracks = np.random.randint(0, num_tracks, size=len(pos_pids))

    X_pids   = np.concatenate([pos_pids,   pos_pids])
    X_tracks = np.concatenate([pos_tracks, neg_tracks])
    y        = np.concatenate([np.ones(len(pos_pids)), np.zeros(len(pos_pids))])

    X_p_train, _, X_t_train, _, y_train, _ = train_test_split(
        X_pids, X_tracks, y, test_size=TEST_SIZE, random_state=42
    )

    model = build_model(num_playlists, num_tracks, EMBEDDING_DIM)
    model.fit(
        [X_p_train, X_t_train], y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
    )

    save_model(model)
    return model

# =============================================================================
# AVALIAÇÃO
# =============================================================================

### --- VARIÁVEIS DE TESTE ---
NUM_PLAYLISTS_TEST = 1
PCT_REMOVED        = 0.20
TOP_K_REC          = 200
# ----------------------------

def evaluate(model: Model, df, interactions_df, pid_map, reverse_track_map, uri_to_name):
    """
    Para cada playlist de teste:
      - Remove PCT_REMOVED% das músicas
      - Gera recomendações com o modelo
      - Imprime acertos (hits) no Top K
    """
    num_tracks = len(reverse_track_map)
    test_pids  = df['pid'].head(NUM_PLAYLISTS_TEST).values

    for pid in test_pids:
        actual_tracks = interactions_df[interactions_df['pid'] == pid]['track_encoded'].values
        n_rem = max(1, int(len(actual_tracks) * PCT_REMOVED))

        removed   = np.random.choice(actual_tracks, n_rem, replace=False)
        remaining = [t for t in actual_tracks if t not in removed]

        print(f"\nPlaylist PID: {pid} | Músicas na Playlist: {len(actual_tracks)}")
        print("Músicas removidas para teste (o que o modelo deve tentar recuperar):")
        for r_idx in removed:
            print(f"  - {uri_to_name[reverse_track_map[r_idx]]}")

        p_idx = pid_map[pid]
        preds = model.predict(
            [np.full(num_tracks, p_idx), np.arange(num_tracks)],
            batch_size=4096, verbose=0
        ).flatten()

        preds[remaining] = -1
        top_indices = preds.argsort()[-TOP_K_REC:][::-1]

        hits = [int(idx) for idx in top_indices if idx in removed]

        print(f"\nRecuperadas no Top {TOP_K_REC}: {len(hits)} de {len(removed)}")
        print("--- Top 30 Recomendações ---")
        for i, idx in enumerate(top_indices[:30]):
            idx_int = int(idx)
            name    = uri_to_name[reverse_track_map[idx_int]]
            status  = "✅ ACERTO" if idx_int in removed else ""
            print(f"{i+1:>3}. {name}  {status}")

# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    df = load_dataset(n_files=200)
    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)

    num_playlists = len(pid_map)
    num_tracks    = len(track_map)

    if model_exists():
        print("Modelo encontrado — carregando do disco...")
        model = load_model_from_disk()
    else:
        print("Nenhum modelo salvo — treinando do zero...")
        model = train(interactions_df, num_playlists, num_tracks)

    evaluate(model, df, interactions_df, pid_map, reverse_track_map, uri_to_name)