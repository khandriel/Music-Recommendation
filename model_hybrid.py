# =============================================================================
# model_hybrid.py — Modelo Híbrido (Colaborativo + Content-Based)
#
# ESTADO ATUAL:
#   Apenas o sinal content-based (TF-IDF + cosseno) está ativo.
#   A integração do colaborativo está estruturada via 'collab_model',
#   mas o método _collab_scores() retorna zeros até ser implementado.
#
# ARQUITETURA PLANEJADA:
#   score_final = collab_weight * score_collab + (1 - collab_weight) * score_content
#
# PARA INTEGRAR O COLABORATIVO (passos futuros):
#   1. Implemente _collab_scores() — veja o bloco TODO nesse método.
#   2. Passe collab_model, pid_map, track_map e reverse_track_map ao construtor.
#   3. Ajuste collab_weight para calibrar o balanço entre os dois sinais.
#
# NOTA SOBRE ALINHAMENTO DE ÍNDICES:
#   - O colaborativo usa track_encoded (índices de track_map em build_interactions).
#   - O content-based usa índices de content_df (build_content_catalog).
#   - Os dois têm orderings distintos. _collab_scores() precisa alinhar via URI.
#
# USO:
#   model = HybridRecommender(content_df, tfidf_matrix, interactions_df, uri_to_name)
#   model.evaluate(pid=None, pct_removed=0.20, top_k=200)
# =============================================================================

import numpy as np
from sklearn.metrics.pairwise import linear_kernel

# =============================================================================
# VARIÁVEIS DE AVALIAÇÃO
# =============================================================================

NUM_PLAYLISTS_TEST_HY = 3
PCT_REMOVED_HY        = 0.20
TOP_K_HY              = 200
COLLAB_WEIGHT_HY      = 0.5   # 0 = só content, 1 = só collab

# =============================================================================
# CLASSE
# =============================================================================

class HybridRecommender:
    """
    Combina filtragem colaborativa e content-based num score único.

    Enquanto o colaborativo não está integrado, opera inteiramente sobre
    similaridade TF-IDF — o collab_weight só passa a ter efeito quando
    _collab_scores() for implementado.
    """

    def __init__(
        self,
        content_df,
        tfidf_matrix,
        interactions_df,
        uri_to_name,
        collab_model=None,
        pid_map=None,
        track_map=None,
        reverse_track_map=None,
        collab_weight: float = COLLAB_WEIGHT_HY,
    ):
        self.content_df      = content_df
        self.tfidf_matrix    = tfidf_matrix
        self.interactions_df = interactions_df
        self.uri_to_name     = uri_to_name

        # Colaborativo — preenchido quando integrado
        self.collab_model      = collab_model
        self.pid_map           = pid_map
        self.track_map         = track_map
        self.reverse_track_map = reverse_track_map

        self.collab_weight = collab_weight

    # ------------------------------------------------------------------
    # Sinal content-based (ativo)
    # ------------------------------------------------------------------

    def _content_scores(self, remaining_uris: list) -> np.ndarray:
        """
        Perfil TF-IDF médio das tracks restantes → cosseno com todo o catálogo.
        Retorna array normalizado em [0, 1] com len == len(content_df).
        """
        idxs = self.content_df[
            self.content_df['track_uri'].isin(remaining_uris)
        ].index

        if len(idxs) == 0:
            return np.zeros(len(self.content_df))

        profile = np.asarray(self.tfidf_matrix[idxs].mean(axis=0))
        scores  = linear_kernel(profile, self.tfidf_matrix).flatten()

        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    # ------------------------------------------------------------------
    # Sinal colaborativo (placeholder)
    # ------------------------------------------------------------------

    def _collab_scores(self, pid, remaining_uris: list) -> np.ndarray:
        """
        Retorna scores colaborativos alinhados ao índice do content_df.

        TODO: implementar quando o modelo colaborativo for integrado.
        Esboço da implementação futura:

            pid_encoded  = self.pid_map[pid]
            num_tracks   = len(self.reverse_track_map)
            all_idxs     = np.arange(num_tracks)

            raw_scores = self.collab_model.predict(
                [np.full(num_tracks, pid_encoded), all_idxs],
                batch_size=4096, verbose=0
            ).flatten()

            # Mapear collab track_encoded → URI → content_df index
            scores_aligned = np.zeros(len(self.content_df))
            for track_enc, score in enumerate(raw_scores):
                uri = self.reverse_track_map[track_enc]
                rows = self.content_df[self.content_df['track_uri'] == uri].index
                if not rows.empty:
                    scores_aligned[rows[0]] = score

            # Normalizar para [0, 1] antes de devolver
            s_min, s_max = scores_aligned.min(), scores_aligned.max()
            if s_max > s_min:
                scores_aligned = (scores_aligned - s_min) / (s_max - s_min)
            return scores_aligned
        """
        return np.zeros(len(self.content_df))

    # ------------------------------------------------------------------
    # Recomendação
    # ------------------------------------------------------------------

    def recommend(
        self,
        pid,
        remaining_uris: list,
        exclude_uris: list = None,
        top_k: int = TOP_K_HY,
    ) -> list:
        """
        Gera top_k URIs recomendadas para uma playlist.

        Parâmetros:
            pid           — PID original da playlist
            remaining_uris — URIs das tracks que permaneceram na playlist
            exclude_uris  — URIs a excluir do ranking (default: remaining_uris)
            top_k         — quantas recomendações retornar
        """
        if exclude_uris is None:
            exclude_uris = list(remaining_uris)

        content_scores = self._content_scores(remaining_uris)
        collab_scores  = self._collab_scores(pid, remaining_uris)

        final_scores = (
            (1 - self.collab_weight) * content_scores
            + self.collab_weight     * collab_scores
        )

        for uri in exclude_uris:
            rows = self.content_df[self.content_df['track_uri'] == uri].index
            if not rows.empty:
                final_scores[rows[0]] = -1.0

        top_indices = final_scores.argsort()[::-1][:top_k]
        return self.content_df.iloc[top_indices]['track_uri'].tolist()

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pid=None,
        num_playlists_test: int = NUM_PLAYLISTS_TEST_HY,
        pct_removed: float      = PCT_REMOVED_HY,
        top_k: int              = TOP_K_HY,
    ) -> float:
        """
        Para cada playlist de teste:
          - Remove pct_removed% das músicas
          - Gera recomendações com o modelo híbrido
          - Imprime acertos e Recall@K

        Retorna o Recall@K geral (em %).
        """
        available_pids = self.interactions_df['pid'].unique()

        if pid is not None:
            test_pids = [pid] if pid in self.interactions_df['pid'].values else [available_pids[0]]
        else:
            test_pids = available_pids[:num_playlists_test]

        total_hits    = 0
        total_removed = 0

        for cur_pid in test_pids:
            all_tracks = self.interactions_df[
                self.interactions_df['pid'] == cur_pid
            ]['track_uri'].values

            if len(all_tracks) == 0:
                print(f"Playlist {cur_pid} não encontrada — pulando.")
                continue

            n_rem     = max(1, int(len(all_tracks) * pct_removed))
            removed   = np.random.choice(all_tracks, n_rem, replace=False)
            remaining = [t for t in all_tracks if t not in removed]

            print(f"\n{'='*60}")
            print(f"Playlist PID: {cur_pid} | Total: {len(all_tracks)} músicas")
            print(f"Removidas: {len(removed)} | Restantes: {len(remaining)}")
            print("Músicas removidas:")
            for uri in removed:
                print(f"  - {self.uri_to_name.get(uri, '?')}")

            rec_uris    = self.recommend(cur_pid, remaining, top_k=top_k)
            removed_set = set(removed.tolist())
            hits        = [u for u in rec_uris if u in removed_set]
            recall      = len(hits) / len(removed) * 100
            total_hits    += len(hits)
            total_removed += len(removed)

            print(f"\nRecuperadas no Top {top_k}: {len(hits)}/{len(removed)} "
                  f"→ Recall@{top_k} = {recall:.1f}%")
            print(f"--- Top 30 Recomendações ---")
            for i, uri in enumerate(rec_uris[:30]):
                row    = self.content_df[self.content_df['track_uri'] == uri]
                name   = row['track_name'].values[0]   if not row.empty else "?"
                artist = row['artist_name'].values[0]  if not row.empty else "?"
                status = "✅ ACERTO" if uri in removed_set else ""
                print(f"{i+1:>3}. {name} — {artist}  {status}")

        overall_recall = total_hits / total_removed * 100 if total_removed > 0 else 0
        print(f"\n{'='*60}")
        print(f"RECALL GERAL @{top_k}: {total_hits}/{total_removed} = {overall_recall:.1f}%")
        return overall_recall
