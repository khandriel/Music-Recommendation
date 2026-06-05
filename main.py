# =============================================================================
# main.py — Ponto de entrada: seleciona e executa o modelo desejado
#
# USO: python main.py
#
# Selecione o modelo alterando a variável MODEL abaixo:
#   "collaborative" — Filtragem Colaborativa pura (NeuMF) — baseline de comparação
#   "content"       — Filtragem Baseada em Conteúdo       — placeholder (a implementar)
#   "hybrid"        — Modelo Híbrido (Content-Based ativo, Colaborativo pendente)
#
# Primeira execução:
#   - Processa os dados e salva em cache/
#   - Se MODEL == "collaborative", treina a rede e salva em saved_models/
# Execuções seguintes:
#   - Carrega o cache de cache/ e o modelo de saved_models/ (se aplicável)
#
# IMPORTANTE: o cache/ NÃO guarda o valor de N_FILES. Se você alterar N_FILES,
# apague a pasta cache/ (e saved_models/) para reprocessar e retreinar.
# =============================================================================

# =============================================================================
# CONFIGURAÇÃO — altere aqui
# =============================================================================
MODEL   = "collaborative"  # "collaborative" | "content" | "hybrid"
N_FILES = 350              # Arquivos JSON a carregar (cada arquivo = 1000 playlists)

# =============================================================================
# IMPORTS
# =============================================================================
# model_content é sempre necessário (gerencia o cache de dados).
# model_collaborative e model_hybrid são importados sob demanda dentro de
# cada bloco, para não carregar TensorFlow/etc. quando não forem usados.

from data_processing import load_dataset, build_interactions, build_content_catalog
import model_content as content

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Dados pré-processados — cache/
    # -------------------------------------------------------------------------
    if content.cache_exists():
        print("Cache encontrado — carregando do disco...")
        print(f"ATENÇÃO: o cache é usado independente de N_FILES (={N_FILES}). "
              "Se mudou N_FILES, apague a pasta cache/ para reprocessar.")
        (interactions_df, pid_map, track_map,
         reverse_track_map, uri_to_name,
         content_df, tfidf, tfidf_matrix) = content.load_cache()
    else:
        print("Nenhum cache encontrado — processando do zero (isso pode demorar)...")
        df = load_dataset(n_files=N_FILES)
        interactions_df, pid_map, track_map, reverse_track_map, uri_to_name = build_interactions(df)
        content_df, tfidf, tfidf_matrix = build_content_catalog(df)
        # O df bruto (playlists + listas de tracks) não é mais necessário —
        # libera a RAM antes de treinar/avaliar.
        del df
        content.save_cache(
            interactions_df, pid_map, track_map, reverse_track_map,
            uri_to_name, content_df, tfidf, tfidf_matrix
        )

    print("\n" + "=" * 60)
    print(f"MODELO SELECIONADO: {MODEL.upper()}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # MODELO: COLLABORATIVE — baseline puro NeuMF
    # -------------------------------------------------------------------------
    if MODEL == "collaborative":
        from model_collaborative import CollaborativeRecommender
        rec = CollaborativeRecommender.load_or_train(
            interactions_df, pid_map, track_map, reverse_track_map, uri_to_name
        )
        rec.evaluate()

    # -------------------------------------------------------------------------
    # MODELO: CONTENT-BASED — placeholder
    # -------------------------------------------------------------------------
    elif MODEL == "content":
        # TODO: instanciar e avaliar o modelo content-based standalone aqui
        raise NotImplementedError(
            "Modelo content-based standalone ainda não implementado.\n"
            "Implemente a lógica em model_content.py e conecte aqui."
        )

    # -------------------------------------------------------------------------
    # MODELO: HYBRID — content-based ativo, colaborativo pendente
    # -------------------------------------------------------------------------
    elif MODEL == "hybrid":
        from model_hybrid import HybridRecommender
        hybrid = HybridRecommender(
            content_df=content_df,
            tfidf_matrix=tfidf_matrix,
            interactions_df=interactions_df,
            uri_to_name=uri_to_name,
            # Descomente as linhas abaixo quando integrar o colaborativo:
            # collab_model=CollaborativeRecommender.load_or_train(
            #     interactions_df, pid_map, track_map, reverse_track_map, uri_to_name
            # ).model,
            # pid_map=pid_map,
            # track_map=track_map,
            # reverse_track_map=reverse_track_map,
            collab_weight=0.5,
        )
        hybrid.evaluate()

    else:
        raise ValueError(
            f"MODEL inválido: '{MODEL}'. Use 'collaborative', 'content' ou 'hybrid'."
        )
