# =============================================================================
# data_processing.py — Carregamento e Pré-processamento dos Dados
#
# REQUISITOS: pip install kaggle kagglehub pandas numpy tqdm
#
# =============================================================================

import kagglehub
import pandas as pd
import numpy as np
import os, glob, json
from tqdm import tqdm

# =============================================================================
# CARREGAMENTO DO DATASET
# =============================================================================

N_FILES = 350  # Número de arquivos JSON a carregar (cada arquivo = 1000 playlists)

def load_dataset(n_files: int = N_FILES) -> pd.DataFrame:
    """
    Baixa (ou usa cache) e carrega N arquivos JSON do dataset.
    Cada arquivo contém 1000 playlists.
    Retorna um DataFrame com todas as playlists carregadas.
    """
    path = kagglehub.dataset_download("himanshuwagh/spotify-million")
    data_path = os.path.join(path, "data")
    # sorted() garante a MESMA ordem de arquivos em qualquer sistema —
    # sem isso, "os primeiros n_files" poderiam variar entre máquinas.
    json_files = sorted(glob.glob(os.path.join(data_path, "*.json")))

    all_playlists = []
    print(f"Carregando {n_files} arquivos JSON...")
    for file in tqdm(json_files[:n_files]):
        with open(file, 'r') as f:
            all_playlists.extend(json.load(f)['playlists'])

    df = pd.DataFrame(all_playlists)
    print(f"Total de playlists carregadas: {len(df)}")
    return df

# =============================================================================
# PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

def build_interactions(df: pd.DataFrame) -> tuple:
    """
    Expande as playlists em pares (playlist, música) e cria os mapeamentos
    de IDs para índices numéricos usados pelos modelos.

    Retorna:
        interactions_df   — DataFrame com colunas pid, track_uri,
                            pid_encoded, track_encoded
        pid_map           — dict {pid -> índice}
        track_map         — dict {track_uri -> índice}
        reverse_track_map — dict {índice -> track_uri}
        uri_to_name       — dict {track_uri -> track_name}

    Otimizações de memória:
      - track_name NÃO é guardado no DataFrame (só vira o dict uri_to_name,
        deduplicado). Nenhum modelo lê a coluna track_name das interações.
      - track_uri vira 'category' (deduplica as strings repetidas internamente).
      - Colunas inteiras usam int32 em vez do int64 padrão.
    """
    # Iterar direto sobre as colunas (zip) é bem mais rápido que df.iterrows().
    # Guardamos só pid e track_uri nas interações; o nome vai para um dict
    # deduplicado, evitando repetir a string em milhões de linhas.
    pids        = []
    uris        = []
    uri_to_name = {}
    for pid, tracks in zip(df['pid'], df['tracks']):
        for track in tracks:
            uri = track['track_uri']
            pids.append(pid)
            uris.append(uri)
            if uri not in uri_to_name:
                uri_to_name[uri] = track['track_name']

    # Constrói as colunas já com o dtype final. Passar listas Python cruas faria
    # o pandas rodar maybe_convert_objects, que aloca um buffer complex128
    # gigante só para inferir o tipo numérico (causa de MemoryError com dados
    # grandes). np.asarray(int32) e pd.Categorical evitam essa inferência.
    interactions_df = pd.DataFrame({
        'pid':       np.asarray(pids, dtype=np.int32),
        'track_uri': pd.Categorical(uris),
    })
    # Libera as listas intermediárias o quanto antes
    del pids, uris

    pid_map           = {id_: i for i, id_ in enumerate(interactions_df['pid'].unique())}
    track_map         = {uri: i for i, uri in enumerate(interactions_df['track_uri'].cat.categories)}
    reverse_track_map = {i: uri for uri, i in track_map.items()}

    interactions_df['pid_encoded']   = interactions_df['pid'].map(pid_map).astype('int32')
    interactions_df['track_encoded'] = interactions_df['track_uri'].map(track_map).astype('int32')

    num_playlists = len(pid_map)
    num_tracks    = len(track_map)
    print(f"Playlists: {num_playlists} | Músicas Únicas: {num_tracks}")

    return interactions_df, pid_map, track_map, reverse_track_map, uri_to_name

def build_content_catalog(df: pd.DataFrame) -> tuple:
    """
    Cria um catálogo de músicas únicas com metadados textuais (nome + artista)
    para uso pelo modelo de Filtragem Baseada em Conteúdo.

    Retorna:
        content_df   — DataFrame com track_uri, track_name, artist_name, metadata
        tfidf        — objeto TfidfVectorizer já treinado
        tfidf_matrix — matriz TF-IDF esparsa (músicas × features)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_unique_tracks = []
    seen_uris = set()

    for tracks in df['tracks']:
        for t in tracks:
            if t['track_uri'] not in seen_uris:
                all_unique_tracks.append({
                    'track_uri':   t['track_uri'],
                    'track_name':  t['track_name'],
                    'artist_name': t['artist_name'],
                    'metadata':    f"{t['track_name']} {t['artist_name']}".lower()
                })
                seen_uris.add(t['track_uri'])

    content_df = pd.DataFrame(all_unique_tracks)

    tfidf        = TfidfVectorizer(stop_words='english', max_features=20000)
    tfidf_matrix = tfidf.fit_transform(content_df['metadata'])

    print(f"Catálogo para Conteúdo: {content_df.shape[0]} músicas | Matriz TF-IDF: {tfidf_matrix.shape}")
    return content_df, tfidf, tfidf_matrix

# =============================================================================
# EXECUÇÃO DIRETA — carrega e pré-processa tudo, imprime resumo
# =============================================================================

if __name__ == "__main__":
    df = load_dataset()
    print(df.head(1).to_string())

    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)

    content_df, tfidf, tfidf_matrix = build_content_catalog(df)
