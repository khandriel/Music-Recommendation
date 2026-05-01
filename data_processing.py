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

def load_dataset(n_files: int = 200) -> pd.DataFrame:
    """
    Baixa (ou usa cache) e carrega N arquivos JSON do dataset.
    Cada arquivo contém 1000 playlists.
    Retorna um DataFrame com todas as playlists carregadas.
    """
    path = kagglehub.dataset_download("himanshuwagh/spotify-million")
    data_path = os.path.join(path, "data")
    json_files = glob.glob(os.path.join(data_path, "*.json"))

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
        interactions_df   — DataFrame com colunas pid, track_uri, track_name,
                            pid_encoded, track_encoded
        pid_map           — dict {pid -> índice}
        track_map         — dict {track_uri -> índice}
        reverse_track_map — dict {índice -> track_uri}
        uri_to_name       — dict {track_uri -> track_name}
    """
    interactions = []
    for _, row in df.iterrows():
        pid = row['pid']
        for track in row['tracks']:
            interactions.append({
                'pid':       pid,
                'track_uri': track['track_uri'],
                'track_name': track['track_name']
            })

    interactions_df = pd.DataFrame(interactions)

    pid_map           = {id_: i for i, id_ in enumerate(interactions_df['pid'].unique())}
    track_map         = {uri: i for i, uri in enumerate(interactions_df['track_uri'].unique())}
    reverse_track_map = {i: uri for uri, i in track_map.items()}
    uri_to_name       = dict(zip(interactions_df['track_uri'], interactions_df['track_name']))

    interactions_df['pid_encoded']   = interactions_df['pid'].map(pid_map)
    interactions_df['track_encoded'] = interactions_df['track_uri'].map(track_map)

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

    for _, row in df.iterrows():
        for t in row['tracks']:
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
    df = load_dataset(n_files=200)
    print(df.head(1).to_string())

    interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)

    content_df, tfidf, tfidf_matrix = build_content_catalog(df)
