# =============================================================================
# main.py — Ponto de entrada: roda ambos os modelos em sequência
#
# USO: python main.py
#
# Primeira execução:
#   - Processa os dados e salva em cache/
#   - Treina o modelo colaborativo e salva em saved_models/
# Execuções seguintes:
#   - Carrega o cache de cache/
#   - Carrega o modelo colaborativo de saved_models/
# =============================================================================

from data_processing    import load_dataset, build_interactions, build_content_catalog
import model_collaborative as collab
import model_content       as content

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Dados pré-processados — cache/
    # -------------------------------------------------------------------------
    if content.cache_exists():
        print("Cache encontrado — carregando do disco...")
        (interactions_df, pid_map, track_map,
         reverse_track_map, uri_to_name,
         content_df, tfidf, tfidf_matrix) = content.load_cache()
        df = None
    else:
        print("Nenhum cache encontrado — processando do zero (isso pode demorar)...")
        df = load_dataset(n_files=200)
        interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)
        content_df, tfidf, tfidf_matrix = build_content_catalog(df)
        content.save_cache(
            interactions_df, pid_map, track_map, reverse_track_map,
            uri_to_name, content_df, tfidf, tfidf_matrix
        )

    num_playlists = len(pid_map)
    num_tracks    = len(track_map)

    # -------------------------------------------------------------------------
    # Modelo 1 — Filtragem Colaborativa — saved_models/
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODELO 1 — FILTRAGEM COLABORATIVA")
    print("=" * 60)

    if collab.model_exists():
        print("Modelo colaborativo encontrado — carregando do disco...")
        cf_model = collab.load_model_from_disk()
    else:
        print("Nenhum modelo salvo — treinando do zero...")
        cf_model = collab.train(interactions_df, num_playlists, num_tracks)

    # df é necessário para evaluate; recarrega se veio do cache
    if df is None:
        df = load_dataset(n_files=200)

    collab.evaluate(cf_model, df, interactions_df, pid_map, reverse_track_map, uri_to_name)

    # -------------------------------------------------------------------------
    # Modelo 2 — Filtragem Baseada em Conteúdo — cache/
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODELO 2 — FILTRAGEM BASEADA EM CONTEÚDO")
    print("=" * 60)

    content.recommend(
        content.PID_TEST_CB, interactions_df, content_df, tfidf_matrix,
        top_k=content.TOP_K_CB
    )
    content.evaluate(
        content.PID_VAL_CB, interactions_df, content_df, tfidf_matrix, uri_to_name,
        pct_removed=content.PCT_REMOVED_CB, top_k=content.TOP_K_VAL_CB
    )