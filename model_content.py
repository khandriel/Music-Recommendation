# =============================================================================
# model_content.py — Filtragem Baseada em Conteúdo (TF-IDF + Cosseno)
#
# REQUISITOS:
#   pip install numpy pandas scikit-learn joblib scipy
#
# USO:
#   Execute diretamente:  python model_content.py
#   Ou importe as funções em outro script:
#       from model_content import recommend, evaluate
#
# NOTA:
#   O modelo content-based não possui pesos treináveis — ele é composto por
#   dados pré-processados (vetorizador TF-IDF, matriz esparsa e DataFrames).
#   Por isso são salvos em cache/ e não em saved_models/.
# =============================================================================

import numpy as np
import joblib
import os
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import linear_kernel

from data_processing import load_dataset, build_interactions, build_content_catalog

# =============================================================================
# CAMINHOS DE PERSISTÊNCIA
# =============================================================================

CACHE_DIR          = "cache"
TFIDF_PATH         = os.path.join(CACHE_DIR, "tfidf_vectorizer.joblib")
TFIDF_MATRIX_PATH  = os.path.join(CACHE_DIR, "tfidf_matrix.npz")
CONTENT_DF_PATH    = os.path.join(CACHE_DIR, "content_df.joblib")
INTERACTIONS_PATH  = os.path.join(CACHE_DIR, "interactions_df.joblib")
MAPS_PATH          = os.path.join(CACHE_DIR, "maps.joblib")

# =============================================================================
# PERSISTÊNCIA — SALVAR E CARREGAR
# =============================================================================

def save_cache(interactions_df, pid_map, track_map, reverse_track_map,
               uri_to_name, content_df, tfidf, tfidf_matrix):
    """Salva os dados pré-processados em cache/ para reutilização."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    joblib.dump(tfidf, TFIDF_PATH)
    save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    joblib.dump(content_df, CONTENT_DF_PATH)
    joblib.dump(interactions_df, INTERACTIONS_PATH)
    joblib.dump({
        "pid_map":           pid_map,
        "track_map":         track_map,
        "reverse_track_map": reverse_track_map,
        "uri_to_name":       uri_to_name,
    }, MAPS_PATH)

    print(f"Cache salvo em '{CACHE_DIR}/'")


def load_cache():
    """
    Carrega os dados pré-processados salvos em cache/.
    Retorna: interactions_df, pid_map, track_map, reverse_track_map,
             uri_to_name, content_df, tfidf, tfidf_matrix
    Lança FileNotFoundError se os arquivos não existirem.
    """
    for path in [TFIDF_PATH, TFIDF_MATRIX_PATH, CONTENT_DF_PATH,
                 INTERACTIONS_PATH, MAPS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache não encontrado: {path}\n"
                "Execute save_cache() primeiro para gerar os arquivos."
            )

    tfidf           = joblib.load(TFIDF_PATH)
    tfidf_matrix    = load_npz(TFIDF_MATRIX_PATH)
    content_df      = joblib.load(CONTENT_DF_PATH)
    interactions_df = joblib.load(INTERACTIONS_PATH)
    maps            = joblib.load(MAPS_PATH)

    print("Cache carregado do disco.")
    return (
        interactions_df,
        maps["pid_map"],
        maps["track_map"],
        maps["reverse_track_map"],
        maps["uri_to_name"],
        content_df,
        tfidf,
        tfidf_matrix,
    )


def cache_exists() -> bool:
    """Retorna True se todos os arquivos de cache já foram salvos."""
    return all(os.path.exists(p) for p in [
        TFIDF_PATH, TFIDF_MATRIX_PATH, CONTENT_DF_PATH,
        INTERACTIONS_PATH, MAPS_PATH
    ])

# =============================================================================
# VARIÁVEIS DE TESTE
# =============================================================================

### --- VARIÁVEIS DE TESTE CONTENT-BASED ---
PID_TEST_CB = None   # None = usa o primeiro PID disponível nos dados
TOP_K_CB    = 30
# ------------------------------------------

### --- VARIÁVEIS DE VALIDAÇÃO CB ---
PID_VAL_CB     = None  # None = usa o primeiro PID disponível nos dados
PCT_REMOVED_CB = 0.20
TOP_K_VAL_CB   = 1000
# ----------------------------------

# =============================================================================
# UTILITÁRIO — resolve PID
# =============================================================================

def _resolve_pid(pid, interactions_df, label="PID"):
    """
    Se pid for None, usa o primeiro PID disponível.
    Se pid não existir nos dados, avisa e usa o primeiro disponível.
    Retorna o pid resolvido.
    """
    available_pids = interactions_df['pid'].unique()

    if pid is None:
        resolved = available_pids[0]
        print(f"{label} não definido — usando PID {resolved}")
        return resolved

    if pid not in interactions_df['pid'].values:
        resolved = available_pids[0]
        print(f"{label} {pid} não encontrado nos dados. Usando PID {resolved} no lugar.")
        return resolved

    return pid

# =============================================================================
# RECOMENDAÇÃO (CONTENT-BASED)
# =============================================================================

def recommend(pid, interactions_df, content_df, tfidf_matrix, top_k: int = 30):
    """
    Gera recomendações para uma playlist usando similaridade de cosseno
    sobre os vetores TF-IDF das músicas.
    Exclui da recomendação as músicas já presentes na playlist.
    """
    pid = _resolve_pid(pid, interactions_df, label="PID_TEST_CB")

    playlist_tracks  = interactions_df[interactions_df['pid'] == pid]['track_uri'].values
    playlist_indices = content_df[content_df['track_uri'].isin(playlist_tracks)].index

    if len(playlist_indices) == 0:
        print(f"Nenhuma música da playlist {pid} encontrada no catálogo de conteúdo.")
        return

    playlist_profile = np.asarray(tfidf_matrix[playlist_indices].mean(axis=0))

    cosine_sim      = linear_kernel(playlist_profile, tfidf_matrix).flatten()
    related_indices = cosine_sim.argsort()[::-1]

    print(f"\n--- TOP {top_k} RECOMENDAÇÕES (CONTENT-BASED) para PID {pid} ---")
    print(f"    Músicas na playlist: {len(playlist_tracks)}")
    count = 0
    for idx in related_indices:
        uri = content_df.iloc[idx]['track_uri']
        if uri not in playlist_tracks:
            name   = content_df.iloc[idx]['track_name']
            artist = content_df.iloc[idx]['artist_name']
            score  = cosine_sim[idx]
            print(f"{count+1:>3}. {name} — {artist}  (score: {score:.4f})")
            count += 1
        if count >= top_k:
            break

# =============================================================================
# VALIDAÇÃO DE RECUPERAÇÃO (CONTENT-BASED)
# =============================================================================

def evaluate(pid, interactions_df, content_df, tfidf_matrix, uri_to_name,
             pct_removed: float = 0.20, top_k: int = 1000):
    """
    Remove PCT_REMOVED% das músicas de uma playlist, reconstrói o perfil
    com o que sobrou e verifica quantas músicas removidas aparecem no Top K.
    """
    pid = _resolve_pid(pid, interactions_df, label="PID_VAL_CB")

    all_playlist_tracks = interactions_df[interactions_df['pid'] == pid]['track_uri'].values

    if len(all_playlist_tracks) == 0:
        print(f"Playlist {pid} não encontrada nos dados.")
        return

    n_rem_cb     = max(1, int(len(all_playlist_tracks) * pct_removed))
    removed_cb   = np.random.choice(all_playlist_tracks, n_rem_cb, replace=False)
    remaining_cb = [t for t in all_playlist_tracks if t not in removed_cb]

    print(f"\nPlaylist PID: {pid} | Total: {len(all_playlist_tracks)} | "
          f"Removidas para teste: {len(removed_cb)}")
    print("Músicas removidas:")
    for uri in removed_cb:
        print(f"  - {uri_to_name.get(uri, 'Desconhecido')}")

    remaining_indices = content_df[content_df['track_uri'].isin(remaining_cb)].index
    if len(remaining_indices) == 0:
        print("Não sobraram músicas suficientes para criar um perfil.")
        return

    val_profile    = np.asarray(tfidf_matrix[remaining_indices].mean(axis=0))
    val_cosine_sim = linear_kernel(val_profile, tfidf_matrix).flatten()

    # Mascarar músicas que já estão na playlist (as que restaram)
    for t_uri in remaining_cb:
        idx_list = content_df[content_df['track_uri'] == t_uri].index
        if not idx_list.empty:
            val_cosine_sim[idx_list[0]] = -1

    top_indices_cb = val_cosine_sim.argsort()[::-1][:top_k]
    rec_uris       = content_df.iloc[top_indices_cb]['track_uri'].values
    hits_cb        = [uri for uri in rec_uris if uri in removed_cb]

    print(f"\nRecuperadas no Top {top_k}: {len(hits_cb)} de {len(removed_cb)}")
    print("--- Top 20 Recomendações ---")
    for i, idx in enumerate(top_indices_cb[:20]):
        uri    = content_df.iloc[idx]['track_uri']
        name   = content_df.iloc[idx]['track_name']
        artist = content_df.iloc[idx]['artist_name']
        status = "✅ ACERTO" if uri in removed_cb else ""
        print(f"{i+1:>3}. {name} — {artist}  {status}")

# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    if cache_exists():
        print("Cache encontrado — carregando do disco...")
        (interactions_df, pid_map, track_map,
         reverse_track_map, uri_to_name,
         content_df, tfidf, tfidf_matrix) = load_cache()
    else:
        print("Nenhum cache encontrado — processando do zero...")
        df = load_dataset(n_files=200)
        interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)
        content_df, tfidf, tfidf_matrix = build_content_catalog(df)
        save_cache(interactions_df, pid_map, track_map, reverse_track_map,
                   uri_to_name, content_df, tfidf, tfidf_matrix)

    recommend(PID_TEST_CB, interactions_df, content_df, tfidf_matrix, top_k=TOP_K_CB)
    evaluate(PID_VAL_CB, interactions_df, content_df, tfidf_matrix, uri_to_name,
             pct_removed=PCT_REMOVED_CB, top_k=TOP_K_VAL_CB)