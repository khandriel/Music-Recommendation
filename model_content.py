# =============================================================================
# model_content.py — Filtragem Baseada em Conteúdo (Content-Based) PURA
#
# ARQUIVO AUTOCONTIDO no mesmo molde dos modelos colaborativos: expõe a classe
# ContentRecommender com o MESMO contrato dos demais —
#
#     ContentRecommender.load_or_train(interactions_df, pid_map, track_map,
#                                      reverse_track_map, uri_to_name)
#
# para ser drop-in na main.py (treino) e no compare_models.py (avaliação).
#
# PURO content-based: NÃO usa co-ocorrência em playlists nem qualquer sinal
# colaborativo. A recomendação vem só do CONTEÚDO da faixa (features de áudio,
# gênero do artista e ano de lançamento).
#
# O QUE É "TREINAR" AQUI: o content-based não aprende pesos como uma rede; o
# "treino" é a ENGENHARIA DE ATRIBUTOS sobre 'audio_features.csv' + 'generos.json'
# (padroniza e normaliza as 10 features de áudio, vetoriza os gêneros em matriz
# esparsa, deriva ano/era/país do ISRC). Esses artefatos são salvos em
# saved_models/content_based.joblib e recarregados nas próximas execuções.
#
# PERSISTÊNCIA (igual aos outros modelos): se o arquivo já existe em
# saved_models/, ele é apenas CARREGADO (não retreina). Para forçar o retreino,
# apague saved_models/content_based.joblib.
#
# LIMPEZA DE RUÍDO NO TREINO: o CSV tem ~2,26M faixas, mas ~42% NÃO possuem
# features de áudio (linhas vazias) — essas são DESCARTADAS (dropna), pois sem
# áudio não há sinal de conteúdo. Faixas sem gênero (artista fora do
# generos.json, ~61%) ou sem ISRC (~0,2%) são MANTIDAS: ainda têm áudio/ano; o
# sinal ausente apenas contribui zero na soma ponderada (descartá-las mataria a
# cobertura do catálogo sem ganho real).
#
# INFERÊNCIA — dois modos, ambos 100% content-based:
#   recomendar(nome_musica, pesos)            -> API nativa (similar a UMA faixa)
#   recommend(pid, seed_idxs, exclude, top_k) -> interface dos colaborativos:
#       representa a playlist pelo PERFIL-CENTROIDE das faixas conhecidas
#       (média do áudio/gênero + ano médio) e ranqueia o catálogo por
#       similaridade de conteúdo a esse perfil. Ponte via URI
#       (track_encoded <-> uri); faixas fora do CSV de áudio ficam de fora.
#
# DADOS NECESSÁRIOS:
#   audio_features.csv  (uri,id,nome,artista,isrc,href + as 10 features) — NÃO
#                        versionado (grande); coloque na raiz do projeto.
#   generos.json        ({artista: [generos]}) — já no repositório.
#
# O ESTUDO DE ABLAÇÃO / FEATURE SELECTION (que precisa das playlists em
# ./archive/data) NÃO roda no import nem no treino: é pesquisa, disponível só
# na execução direta deste arquivo, atrás da flag RUN_ABLATION.
#
# USO:
#   Treinar:   python model_content.py     (ou via main.py)
#   Comparar:  python compare_models.py
# =============================================================================

import os
# BLAS em 1 thread ANTES de importar numpy — consistente com os demais modelos
# (evita oversubscription quando vários processos rodam em sequência).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json as _json
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, normalize
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'tempo',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'key', 'mode',
]

SIGMA_ANO = 15   # desvio padrão do kernel Gaussiano de ano (anos)

CSV_PATH     = 'audio_features.csv'
GENEROS_PATH = 'generos.json'
CSV_CHUNK    = 200_000   # leitura em blocos: poupa RAM no CSV de ~390MB

MODELS_DIR = "saved_models"
MODEL_PATH = os.path.join(MODELS_DIR, "content_based.joblib")

# Pesos default da soma ponderada de scores (não precisam somar 1). São os três
# sinais de conteúdo dominantes; áudio pesa mais, gênero e ano complementam.
PESOS_DEFAULT = {'audio': 0.70, 'genero': 0.20, 'ano': 0.10}


def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


# =============================================================================
# HELPERS DE ENGENHARIA DE ATRIBUTOS (usados no treino)
# =============================================================================

def _isrc_para_ano(isrc):
    """Extrai o ano de lançamento dos 2 dígitos do ISRC (posições 6-7)."""
    try:
        s = str(isrc).strip().replace('-', '')
        if len(s) < 7:
            return np.nan
        yy = int(s[5:7])
        return 2000 + yy if yy <= 25 else 1900 + yy
    except Exception:
        return np.nan


def _era_musical(ano):
    if   ano < 1970: return 'pre_70'
    elif ano < 1980: return 'anos_70'
    elif ano < 1990: return 'anos_80'
    elif ano < 2000: return 'anos_90'
    elif ano < 2010: return 'anos_2000'
    elif ano < 2020: return 'anos_2010'
    else:            return 'anos_2020'


# =============================================================================
# MODELO
# =============================================================================

class ContentRecommender:
    """Content-based puro: soma ponderada de similaridades (áudio, gênero, ano...)."""

    def __init__(self, df, mat_audio, mat_generos,
                 track_map=None, reverse_track_map=None):
        # df DEVE estar com índice 0..N-1 alinhado às linhas das matrizes.
        self.df          = df.reset_index(drop=True)
        self.mat_audio   = np.asarray(mat_audio, dtype=np.float32)  # (N, 10) L2-norm
        self.mat_generos = mat_generos.tocsr()                      # (N, G) esparsa L2-norm
        self.N           = len(self.df)

        # Arrays pré-extraídos (evita .loc/.values repetido nos scores)
        self.uris      = self.df['uri'].values
        self.artistas  = self.df['artista'].values
        self.anos      = self.df['ano'].values.astype(np.float32)
        self.eras      = self.df['era'].values
        self.paises    = self.df['pais_isrc'].values
        self.nomes_low = self.df['nome'].astype(str).str.lower()

        self.uri_to_cidx = {u: i for i, u in enumerate(self.uris)}

        self.score_funcs = {
            'audio':   self.score_audio,
            'genero':  self.score_genero,
            'artista': self.score_artista,
            'ano':     self.score_ano,
            'era':     self.score_era,
            'pais':    self.score_pais,
        }

        # Ponte para a interface colaborativa (track_encoded <-> uri). Vêm dos
        # dados do data_processing, NÃO do modelo salvo (dependem do N_FILES).
        self._set_bridge(track_map, reverse_track_map)

    def _set_bridge(self, track_map, reverse_track_map):
        self.track_map         = track_map
        self.reverse_track_map = reverse_track_map
        if track_map is None:
            self.num_tracks = self.cidx_to_enc = self.enc_to_cidx = None
            return
        self.num_tracks = len(track_map)
        # linha do catálogo de conteúdo -> track_encoded (ou -1 se desconhecida)
        self.cidx_to_enc = np.fromiter(
            (track_map.get(u, -1) for u in self.uris),
            dtype=np.int64, count=self.N)
        # track_encoded -> linha do catálogo (ou -1 se a faixa não tem áudio)
        self.enc_to_cidx = np.full(self.num_tracks, -1, dtype=np.int64)
        valido = self.cidx_to_enc >= 0
        self.enc_to_cidx[self.cidx_to_enc[valido]] = np.nonzero(valido)[0]

    # --------------------------------------------------------------------- #
    # FACTORY — mesmo contrato dos colaborativos
    # --------------------------------------------------------------------- #
    @classmethod
    def load_or_train(cls, interactions_df=None, pid_map=None, track_map=None,
                      reverse_track_map=None, uri_to_name=None):
        """
        Assinatura idêntica aos modelos colaborativos (drop-in na main.py e no
        compare_models.py). interactions_df, pid_map e uri_to_name NÃO são usados
        (o content treina a partir do CSV); track_map/reverse_track_map só
        servem à ponte usada pelo recommend() colaborativo.
        """
        if model_exists():
            print(f"[Content] Modelo encontrado — carregando de '{MODEL_PATH}'...")
            self = cls.load(track_map, reverse_track_map)
        else:
            print("[Content] Nenhum modelo salvo — treinando do zero...")
            self = cls.train(track_map, reverse_track_map)
            self.save()
            print(f"[Content] Modelo salvo em '{MODEL_PATH}'")
        return self

    # --------------------------------------------------------------------- #
    # TREINO — engenharia de atributos a partir do CSV + generos.json
    # --------------------------------------------------------------------- #
    @classmethod
    def train(cls, track_map=None, reverse_track_map=None):
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"[Content] '{CSV_PATH}' não encontrado. O content-based precisa "
                f"do CSV de features de áudio (uma faixa por linha com colunas "
                f"uri,nome,artista,isrc + {AUDIO_FEATURES}). Coloque-o na raiz "
                f"do projeto e rode novamente."
            )

        # ── Leitura em blocos + dropna por bloco (poupa RAM; descarta as ~42%
        #    de faixas SEM features de áudio, que não têm sinal de conteúdo) ──
        usecols = ['uri', 'nome', 'artista', 'isrc'] + AUDIO_FEATURES
        partes = []
        total = 0
        for ch in pd.read_csv(CSV_PATH, usecols=usecols,
                              dtype={'uri': str, 'nome': str,
                                     'artista': str, 'isrc': str},
                              chunksize=CSV_CHUNK):
            total += len(ch)
            partes.append(ch.dropna(subset=AUDIO_FEATURES))
        df = pd.concat(partes, ignore_index=True)
        del partes
        # Segurança: 1 linha por URI e sem URIs nulas (o CSV atual já é limpo).
        df = df.dropna(subset=['uri']).drop_duplicates('uri', keep='first')
        df = df.reset_index(drop=True)
        print(f"[Content] Faixas: {len(df):,} com áudio "
              f"(descartadas {total - len(df):,} sem features, de {total:,})")

        # ── Ano e Era via ISRC ────────────────────────────────────────────
        anos = df['isrc'].apply(_isrc_para_ano)
        df['ano'] = anos.fillna(anos.median())
        df['era'] = df['ano'].apply(_era_musical)
        print(f"[Content] Ano: {int(df['ano'].min())}–{int(df['ano'].max())} "
              f"| sem ISRC: {anos.isna().sum():,}")

        # ── País via ISRC (grupos raros viram 'OUTRO') ────────────────────
        df['pais_isrc'] = df['isrc'].str[:2].str.upper().fillna('XX')
        validos = df['pais_isrc'].value_counts()[lambda x: x >= len(df) * 0.001].index
        df['pais_isrc'] = df['pais_isrc'].where(df['pais_isrc'].isin(validos), 'OUTRO')
        print(f"[Content] Países ISRC: {df['pais_isrc'].nunique()} grupos")

        # ── Features de áudio: padroniza e normaliza L2 (cosine = dot) ─────
        scaler = StandardScaler()
        mat_std = scaler.fit_transform(df[AUDIO_FEATURES]).astype(np.float32)
        normas = np.linalg.norm(mat_std, axis=1, keepdims=True)
        normas[normas == 0] = 1
        mat_audio = (mat_std / normas)   # (N, 10)

        # ── Gêneros via generos.json (matriz esparsa para poupar RAM) ──────
        try:
            with open(GENEROS_PATH, 'r', encoding='utf-8') as f:
                dict_generos = _json.load(f)
        except FileNotFoundError:
            print(f"[Content] '{GENEROS_PATH}' não encontrado — gêneros vazios.")
            dict_generos = {}

        df['generos'] = df['artista'].map(lambda a: dict_generos.get(a, []))
        sem_genero = df['generos'].map(len).eq(0).sum()
        mlb = MultiLabelBinarizer(sparse_output=True)
        mat_gen_bin = mlb.fit_transform(df['generos']).astype(np.float32)
        mat_generos = normalize(mat_gen_bin, norm='l2', axis=1)
        print(f"[Content] Gêneros únicos: {len(mlb.classes_):,} "
              f"| faixas sem gênero (sinal=0): {sem_genero:,} ({sem_genero/len(df):.0%})")

        # Mantém só as colunas usadas em score/recomendação (slim para salvar)
        df = df[['uri', 'nome', 'artista', 'ano', 'era', 'pais_isrc', 'generos']]

        return cls(df, mat_audio, mat_generos, track_map, reverse_track_map)

    # --------------------------------------------------------------------- #
    # PERSISTÊNCIA
    # --------------------------------------------------------------------- #
    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Salva apenas o catálogo de conteúdo (independe do N_FILES colaborativo).
        joblib.dump({
            'df':          self.df,
            'mat_audio':   self.mat_audio,
            'mat_generos': self.mat_generos,
        }, MODEL_PATH)

    @classmethod
    def load(cls, track_map=None, reverse_track_map=None):
        payload = joblib.load(MODEL_PATH)
        return cls(payload['df'], payload['mat_audio'], payload['mat_generos'],
                   track_map, reverse_track_map)

    # --------------------------------------------------------------------- #
    # SCORES INDIVIDUAIS (por faixa) — cada um recebe o índice da faixa-alvo e
    # devolve um array (N,) em [0, 1] com a similaridade a todas as faixas.
    # --------------------------------------------------------------------- #
    def score_audio(self, idx: int) -> np.ndarray:
        s = cosine_similarity(self.mat_audio[idx:idx + 1], self.mat_audio).flatten()
        return np.clip((s + 1) / 2, 0, 1)

    def score_genero(self, idx: int) -> np.ndarray:
        s = cosine_similarity(self.mat_generos[idx:idx + 1], self.mat_generos).flatten()
        return np.clip(s, 0, 1)

    def score_artista(self, idx: int) -> np.ndarray:
        return (self.artistas == self.artistas[idx]).astype(np.float32)

    def score_ano(self, idx: int, sigma: float = SIGMA_ANO) -> np.ndarray:
        return np.exp(-0.5 * ((self.anos - self.anos[idx]) / sigma) ** 2).astype(np.float32)

    def score_era(self, idx: int) -> np.ndarray:
        return (self.eras == self.eras[idx]).astype(np.float32)

    def score_pais(self, idx: int) -> np.ndarray:
        return (self.paises == self.paises[idx]).astype(np.float32)

    def _score_combinado(self, idx: int, pesos: dict) -> np.ndarray:
        """Soma ponderada (pesos normalizados) dos scores ativos para a faixa idx."""
        total = sum(v for v in pesos.values() if v > 0) or 1.0
        score = np.zeros(self.N, dtype=np.float32)
        for feat, peso in pesos.items():
            if peso > 0:
                score += (peso / total) * self.score_funcs[feat](idx)
        return score

    # --------------------------------------------------------------------- #
    # INFERÊNCIA NATIVA — por nome de música (similar a UMA faixa)
    # --------------------------------------------------------------------- #
    def recomendar(self, nome_musica: str, pesos: dict = None, top_k: int = 20,
                   excluir_mesmo_artista: bool = False) -> pd.DataFrame:
        """Top-k músicas mais similares a `nome_musica` (busca por substring)."""
        pesos = pesos or PESOS_DEFAULT
        mascara = self.nomes_low.str.contains(nome_musica.lower(), na=False)
        if not mascara.any():
            print(f"'{nome_musica}' não encontrada.")
            return pd.DataFrame()

        if mascara.sum() > 1:
            print(f" {mascara.sum()} músicas encontradas — usando a primeira.")
        idx_alvo = self.df[mascara].index[0]
        artista_alvo = self.artistas[idx_alvo]

        score = self._score_combinado(idx_alvo, pesos)
        score[idx_alvo] = -1
        if excluir_mesmo_artista:
            score[self.artistas == artista_alvo] = -1

        top_idx = np.argsort(score)[::-1][:top_k]
        resultado = self.df.loc[
            top_idx, ['nome', 'artista', 'ano', 'era', 'generos', 'pais_isrc']
        ].copy()
        resultado['score'] = score[top_idx].round(4)
        resultado['ano'] = resultado['ano'].astype(int)
        resultado['generos'] = resultado['generos'].apply(
            lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '')
            if isinstance(x, list) and x else 'Nenhum')

        row = self.df.loc[idx_alvo]
        gen_alvo = ', '.join(row['generos']) if row['generos'] else 'Desconhecido'
        print(f"\n[Content] Top {top_k} para: '{row['nome']}' - {row['artista']} "
              f"({int(row['ano'])}) | Generos: {gen_alvo}")
        return resultado.reset_index(drop=True)

    # --------------------------------------------------------------------- #
    # INFERÊNCIA COLABORATIVA-COMPATÍVEL — mesma assinatura dos colaborativos
    # (drop-in p/ compare_models). PURO content-based: a playlist é o
    # PERFIL-CENTROIDE das faixas conhecidas (média do áudio/gênero + ano médio)
    # e o catálogo é ranqueado por similaridade de conteúdo a esse perfil.
    # Ponte via URI: faixas sem áudio (fora do CSV) não entram no ranking.
    # --------------------------------------------------------------------- #
    def _centroid_scores(self, cidx: np.ndarray, pesos: dict) -> np.ndarray:
        """Score de conteúdo (N,) do catálogo contra o centroide das faixas cidx."""
        score = np.zeros(self.N, dtype=np.float32)
        if len(cidx) == 0:
            return score
        # No perfil-centroide só os sinais vetorizáveis fazem sentido (áudio,
        # gênero, ano); pesos de era/país/artista são ignorados aqui.
        usados = {f: w for f, w in pesos.items()
                  if w > 0 and f in ('audio', 'genero', 'ano')}
        total = sum(usados.values()) or 1.0

        if 'audio' in usados:
            prof = self.mat_audio[cidx].mean(axis=0)          # (10,)
            s = self.mat_audio @ prof                         # (N,) média dos cossenos
            s = np.clip((s + 1.0) / 2.0, 0.0, 1.0)
            score += (usados['audio'] / total) * s

        if 'genero' in usados:
            prof = np.asarray(self.mat_generos[cidx].mean(axis=0)).ravel()  # (G,)
            s = np.asarray(self.mat_generos @ prof).ravel()   # (N,)
            s = np.clip(s, 0.0, 1.0)
            score += (usados['genero'] / total) * s

        if 'ano' in usados:
            ano_med = float(self.anos[cidx].mean())
            s = np.exp(-0.5 * ((self.anos - ano_med) / SIGMA_ANO) ** 2).astype(np.float32)
            score += (usados['ano'] / total) * s

        return score

    def score(self, pid_encoded, seed_idxs, pesos: dict = None):
        """
        Vetor de scores de conteúdo (num_tracks,) do catálogo contra o perfil-
        centroide das seeds, no espaço colaborativo (track_encoded), SEM aplicar
        exclusões. Base comum de recommend() e do híbrido (late fusion). Faixas
        sem features de áudio (fora do CSV) ficam -inf: o conteúdo se ABSTÉM
        delas — o híbrido trata isso usando só o colaborativo nessas faixas.
        """
        if self.track_map is None or self.reverse_track_map is None:
            raise RuntimeError(
                "score()/recommend() colaborativo precisa de track_map/"
                "reverse_track_map — passe os dados do data_processing em "
                "load_or_train().")

        pesos = pesos or PESOS_DEFAULT
        seed_idxs = np.asarray(seed_idxs, dtype=np.int64)

        # seeds (track_encoded) -> linhas do catálogo (descarta as sem áudio)
        cidx = self.enc_to_cidx[seed_idxs] if len(seed_idxs) else np.empty(0, np.int64)
        cidx = cidx[cidx >= 0]

        scores_cat = self._centroid_scores(cidx, pesos)       # (N,) no espaço do catálogo

        # Catálogo -> espaço colaborativo (track_encoded). Faixas que o
        # colaborativo não conhece ficam de fora; as não cobertas ficam -inf.
        scores = np.full(self.num_tracks, -np.inf, dtype=np.float64)
        valido = self.cidx_to_enc >= 0
        scores[self.cidx_to_enc[valido]] = scores_cat[valido]
        return scores

    def recommend(self, pid_encoded, seed_idxs, exclude_idxs, top_k=500,
                  pesos: dict = None):
        scores = self.score(pid_encoded, seed_idxs, pesos)

        exclude_idxs = np.asarray(exclude_idxs, dtype=np.int64)
        if len(exclude_idxs) > 0:
            scores[exclude_idxs] = -np.inf

        k = min(top_k, scores.shape[0])
        part = np.argpartition(scores, -k)[-k:]
        return part[np.argsort(scores[part])[::-1]].tolist()


# =============================================================================
# PESQUISA — Ground Truth + Feature Selection (ablação). NÃO roda no treino;
# só na execução direta deste arquivo com RUN_ABLATION=True (precisa das
# playlists em ./archive/data).
# =============================================================================

def precision_at_k(top_uris, relevantes, k):
    return sum(1 for u in top_uris[:k] if u in relevantes) / k if k else 0.0


def recall_at_k(top_uris, relevantes, k):
    if not relevantes:
        return 0.0
    return len(set(top_uris[:k]) & relevantes) / len(relevantes)


def ndcg_at_k(top_uris, relevantes, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, u in enumerate(top_uris[:k]) if u in relevantes)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevantes), k)))
    return dcg / idcg if idcg > 0 else 0.0


def _build_track_playlists(pasta_json='./archive/data'):
    """Índice URI -> set de playlists (co-ocorrência), para o ground truth."""
    import glob
    from tqdm import tqdm
    arquivos = glob.glob(os.path.join(pasta_json, '**', '*.json'), recursive=True)
    print(f"\n{len(arquivos)} arquivos JSON de playlists em '{pasta_json}'")
    track_playlists = {}
    for arq in tqdm(arquivos, desc="Indexando"):
        with open(arq, encoding='utf-8') as f:
            try:
                data = _json.load(f)
            except Exception:
                continue
        for pl in data.get('playlists', []):
            pid = pl['pid']
            for t in pl.get('tracks', []):
                track_playlists.setdefault(t['track_uri'], set()).add(pid)
    print(f"Índice: {len(track_playlists):,} URIs únicas")
    return track_playlists


def feature_selection_ablation(rec: ContentRecommender,
                               pasta_json='./archive/data',
                               output_csv='feature_selection.csv'):
    """Estudo de ablação: avalia cada combinação de features contra o ground
    truth de co-ocorrência. Pesado e dependente de ./archive/data."""
    from itertools import product
    from tqdm import tqdm

    ALL_FEATURES  = ['audio', 'genero', 'artista', 'ano', 'era', 'pais']
    K_VALUES      = [5, 10, 20]
    N_AMOSTRAS    = 300
    MIN_PLAYLISTS = 50
    MIN_COOC      = 2
    SEED          = 42

    track_playlists = _build_track_playlists(pasta_json)
    df = rec.df
    uri_series = df['uri']

    def get_coocorrencias(uri_alvo, min_cooc=MIN_COOC):
        pls = track_playlists.get(uri_alvo, set())
        if not pls:
            return set()
        return {u for u in uri_series if u != uri_alvo
                and len(pls & track_playlists.get(u, set())) >= min_cooc}

    uri_set = set(track_playlists.keys())
    df_elegivel = df[
        df['uri'].isin(uri_set) &
        df['uri'].map(lambda u: len(track_playlists.get(u, set())) >= MIN_PLAYLISTS)
    ].copy()
    amostra = df_elegivel.sample(n=min(N_AMOSTRAS, len(df_elegivel)), random_state=SEED)

    combinacoes = []
    for bits in product([0, 1], repeat=len(ALL_FEATURES)):
        config = dict(zip(ALL_FEATURES, bits))
        if config['audio'] == 0 and config['genero'] == 0:
            continue
        if sum(bits) == 0:
            continue
        combinacoes.append(config)

    print(f"\nMúsicas elegíveis : {len(df_elegivel):,}")
    print(f"Amostra           : {len(amostra):,}")
    print(f"Combinações válidas: {len(combinacoes)}\n")

    max_k = max(K_VALUES)
    resultados = []
    for config in tqdm(combinacoes, desc="Ablation"):
        features_ativas = [f for f, v in config.items() if v == 1]
        peso_unitario   = 1.0 / len(features_ativas)
        acum   = {k: {'p': [], 'r': [], 'n': []} for k in K_VALUES}
        sem_gt = 0
        for _, row in amostra.iterrows():
            idx_alvo = row.name
            gt = get_coocorrencias(row['uri'])
            if not gt:
                sem_gt += 1
                continue
            score = np.zeros(len(df), dtype=np.float32)
            for feat in features_ativas:
                score += peso_unitario * rec.score_funcs[feat](idx_alvo)
            score[idx_alvo] = -1
            top_uris = df.loc[np.argsort(score)[::-1][:max_k], 'uri'].tolist()
            for k in K_VALUES:
                acum[k]['p'].append(precision_at_k(top_uris, gt, k))
                acum[k]['r'].append(recall_at_k(top_uris, gt, k))
                acum[k]['n'].append(ndcg_at_k(top_uris, gt, k))

        linha = {f: v for f, v in config.items()}
        linha['features_ativas'] = '+'.join(features_ativas)
        linha['n_features']      = len(features_ativas)
        linha['sem_gt']          = sem_gt
        linha['avaliadas']       = len(amostra) - sem_gt
        for k in K_VALUES:
            n = len(acum[k]['p'])
            linha[f'precision@{k}'] = round(np.mean(acum[k]['p']), 5) if n else 0.0
            linha[f'recall@{k}']    = round(np.mean(acum[k]['r']), 5) if n else 0.0
            linha[f'ndcg@{k}']      = round(np.mean(acum[k]['n']), 5) if n else 0.0
        resultados.append(linha)

    df_sel = pd.DataFrame(resultados)
    df_sel.to_csv(output_csv, index=False)
    print(f"\n{len(df_sel)} combinações avaliadas → {output_csv}")

    best = df_sel.loc[df_sel['ndcg@10'].idxmax()]
    print("\n" + "=" * 60)
    print("MELHOR COMBINAÇÃO DE FEATURES")
    print(f"  Features ativas : {best['features_ativas']}")
    print(f"  NDCG@10         : {best['ndcg@10']:.5f}")
    print("=" * 60)
    return df_sel


# =============================================================================
# EXECUÇÃO DIRETA — apenas TREINA (mesmo padrão dos colaborativos).
# Defina RUN_ABLATION=True para rodar o estudo de feature selection (lento e
# dependente de ./archive/data).
# =============================================================================

RUN_ABLATION = False

if __name__ == "__main__":
    rec = ContentRecommender.load_or_train()
    print("[Content] Concluído. Para avaliar/comparar: python compare_models.py")

    # Exemplo de recomendação por nome (API nativa do content-based)
    r = rec.recomendar("Billie Jean", PESOS_DEFAULT, top_k=15,
                       excluir_mesmo_artista=True)
    if not r.empty:
        print(r.to_string(index=False))

    if RUN_ABLATION:
        feature_selection_ablation(rec)
