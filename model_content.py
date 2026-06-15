# =============================================================================
# 1. Imports
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, normalize
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import glob, os, json as _json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

AUDIO_FEATURES = [
    'danceability', 'energy', 'valence', 'tempo',
    'instrumentalness', 'liveness', 'loudness',
    'speechiness', 'key', 'mode'
]

SIGMA_ANO = 15   # desvio padrão do kernel Gaussiano (anos)

print("OK")


# =============================================================================
# 2. Carregamento e Feature Engineering
# =============================================================================
CSV_PATH = 'audio_features.csv'

df_raw = pd.read_csv(CSV_PATH, dtype={
    'uri':str,'id':str,'nome':str,'artista':str,'isrc':str
})
df = df_raw.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)
print(f"Dataset: {len(df):,} músicas com features de áudio")

# ── Ano e Era via ISRC ───────────────────────────────────────────────
def isrc_para_ano(isrc):
    try:
        s = str(isrc).strip().replace('-', '')
        if len(s) < 7: return np.nan
        yy = int(s[5:7])
        return 2000 + yy if yy <= 25 else 1900 + yy
    except:
        return np.nan

anos = df['isrc'].apply(isrc_para_ano)
df['ano'] = anos.fillna(anos.median())

def era_musical(ano):
    if   ano < 1970: return 'pre_70'
    elif ano < 1980: return 'anos_70'
    elif ano < 1990: return 'anos_80'
    elif ano < 2000: return 'anos_90'
    elif ano < 2010: return 'anos_2000'
    elif ano < 2020: return 'anos_2010'
    else:            return 'anos_2020'

df['era'] = df['ano'].apply(era_musical)
print(f"Ano: {int(df['ano'].min())} – {int(df['ano'].max())} | sem ISRC: {anos.isna().sum():,}")

# ── País via ISRC ────────────────────────────────────────────────────
df['pais_isrc'] = df['isrc'].str[:2].str.upper().fillna('XX')
paises_validos  = df['pais_isrc'].value_counts()[lambda x: x >= len(df) * 0.001].index
df['pais_isrc'] = df['pais_isrc'].where(df['pais_isrc'].isin(paises_validos), 'OUTRO')
print(f"Países ISRC: {df['pais_isrc'].nunique()} grupos")

# ── Feature de Áudio ─────────────────────────────────────────────────
scaler_audio  = StandardScaler()
mat_audio_std = scaler_audio.fit_transform(df[AUDIO_FEATURES]).astype(np.float32)

# Normaliza L2 para cosine similarity eficiente
normas_audio  = np.linalg.norm(mat_audio_std, axis=1, keepdims=True)
normas_audio[normas_audio == 0] = 1
MAT_AUDIO = (mat_audio_std / normas_audio)   # shape (N, 10)

# ── Gênero Musical via generos.json ──────────────────────────────────
print("Carregando gêneros do arquivo generos.json...")
try:
    with open('generos.json', 'r', encoding='utf-8') as f:
        dict_generos = _json.load(f)
except FileNotFoundError:
    print(" Arquivo 'generos.json' não encontrado. Usando listas vazias.")
    dict_generos = {}

df['generos'] = df['artista'].map(lambda a: dict_generos.get(a, []))

print("Vetorizando gêneros (usando Matriz Esparsa para poupar RAM)...")
# A MÁGICA ESTÁ AQUI: sparse_output=True
mlb = MultiLabelBinarizer(sparse_output=True)

# Gera a matriz esparsa (isso vai rodar em segundos e gastar pouca RAM)
mat_generos_bin = mlb.fit_transform(df['generos']).astype(np.float32)

# Usa o normalize do próprio sklearn, que é otimizado para matrizes esparsas!
MAT_GENEROS = normalize(mat_generos_bin, norm='l2', axis=1)

print(f"Total de gêneros únicos mapeados: {len(mlb.classes_)}")


# =============================================================================
# 3. Funções de Score Individuais
# Cada função recebe o índice da música alvo e retorna um array (N,) em [0, 1].
# =============================================================================
def score_audio(idx: int) -> np.ndarray:
    """Cosine similarity sobre as 10 features de áudio. Mapeado para [0,1]."""
    s = cosine_similarity(MAT_AUDIO[idx:idx+1], MAT_AUDIO).flatten()
    return np.clip((s + 1) / 2, 0, 1)

def score_genero(idx: int) -> np.ndarray:
    """Cosine similarity (sobreposição) das listas de gêneros reais do artista. [0,1]"""
    # A matriz já é positiva, o resultado do cosseno será entre 0 e 1.
    s = cosine_similarity(MAT_GENEROS[idx:idx+1], MAT_GENEROS).flatten()
    return np.clip(s, 0, 1)

def score_artista(idx: int) -> np.ndarray:
    """1.0 se mesmo artista, 0.0 caso contrário."""
    a = df.loc[idx, 'artista']
    return (df['artista'].values == a).astype(np.float32)

def score_ano(idx: int, sigma: float = SIGMA_ANO) -> np.ndarray:
    """Kernel Gaussiano: 1.0 no mesmo ano, cai conforme a distância aumenta."""
    ano_alvo = df.loc[idx, 'ano']
    return np.exp(-0.5 * ((df['ano'].values - ano_alvo) / sigma) ** 2).astype(np.float32)

def score_era(idx: int) -> np.ndarray:
    """1.0 se mesma era (décade), 0.0 caso contrário."""
    e = df.loc[idx, 'era']
    return (df['era'].values == e).astype(np.float32)

def score_pais(idx: int) -> np.ndarray:
    """1.0 se mesmo país de origem ISRC, 0.0 caso contrário."""
    p = df.loc[idx, 'pais_isrc']
    return (df['pais_isrc'].values == p).astype(np.float32)

SCORE_FUNCS = {
    'audio':   score_audio,
    'genero':  score_genero,
    'artista': score_artista,
    'ano':     score_ano,
    'era':     score_era,
    'pais':    score_pais,
}

print("Score functions:", list(SCORE_FUNCS.keys()))


# =============================================================================
# 4. Recomendador
# =============================================================================
def recomendar(
    nome_musica: str,
    pesos: dict,
    top_k: int = 20,
    excluir_mesmo_artista: bool = False,
) -> pd.DataFrame:
    """
    Retorna top_k músicas mais similares.
    pesos : dict  feature → peso  (não precisam somar 1)
    """
    mascara = df['nome'].str.lower().str.contains(nome_musica.lower(), na=False)
    if not mascara.any():
        print(f"'{nome_musica}' não encontrada."); return pd.DataFrame()

    n_hits = mascara.sum()
    if n_hits > 1:
        print(f" {n_hits} músicas encontradas — usando a primeira.")

    idx_alvo = df[mascara].index[0]
    row_alvo = df.loc[idx_alvo]

    total = sum(v for v in pesos.values() if v > 0)
    pesos_norm = {k: v / total for k, v in pesos.items() if v > 0}

    score = np.zeros(len(df), dtype=np.float32)
    for feat, peso in pesos_norm.items():
        score += peso * SCORE_FUNCS[feat](idx_alvo)

    score[idx_alvo] = -1
    if excluir_mesmo_artista:
        score[df['artista'].values == row_alvo['artista']] = -1

    top_idx  = np.argsort(score)[::-1][:top_k]
    
    resultado = df.loc[top_idx, ['nome','artista','ano','era','generos','pais_isrc']].copy()
    resultado.insert(len(resultado.columns), 'score', score[top_idx].round(4))
    resultado['ano'] = resultado['ano'].astype(int)
    
    # Formata a lista de gêneros para string limpa no DataFrame final
    resultado['generos'] = resultado['generos'].apply(lambda x: ', '.join(x[:3]) + ('...' if len(x)>3 else '') if isinstance(x, list) and x else 'Nenhum')

    str_generos_alvo = ', '.join(row_alvo['generos']) if row_alvo['generos'] else 'Desconhecido'
    print(f"\n🎵 Top {top_k} para: '{row_alvo['nome']}' — {row_alvo['artista']} "
          f"({int(row_alvo['ano'])}) | Gêneros: {str_generos_alvo}")
    print("-" * 72)
    print(f"Pesos normalizados: { {k: round(v,3) for k,v in pesos_norm.items()} }")
    return resultado.reset_index(drop=True)

print("recomendar() pronta.")


# =============================================================================
# 5. Exemplos
# =============================================================================
PESOS_DEFAULT = {'audio': 0.70, 'genero': 0.20, 'ano': 0.10}

r = recomendar("Billie Jean", PESOS_DEFAULT, top_k=20, excluir_mesmo_artista=True)
print(r.to_string(index=False))

r = recomendar("Billie Jean", {'audio': 0.80, 'genero': 0.20}, top_k=15, excluir_mesmo_artista=True)
print(r.to_string(index=False))


# =============================================================================
# 6. Ground Truth — Co-ocorrência em Playlists
# =============================================================================
PASTA_JSON = './archive/data'   

arquivos_json = glob.glob(os.path.join(PASTA_JSON, '**', '*.json'), recursive=True)
print(f"\n{len(arquivos_json)} arquivos JSON de playlists")

track_playlists: dict = {}
for arq in tqdm(arquivos_json, desc="Indexando"):
    with open(arq, encoding='utf-8') as f:
        try: data = _json.load(f)
        except: continue
    for pl in data.get('playlists', []):
        pid = pl['pid']
        for t in pl.get('tracks', []):
            track_playlists.setdefault(t['track_uri'], set()).add(pid)

print(f"Índice: {len(track_playlists):,} URIs únicas")


# =============================================================================
# 7. Métricas
# =============================================================================
def precision_at_k(top_uris, relevantes, k):
    return sum(1 for u in top_uris[:k] if u in relevantes) / k if k else 0.0

def recall_at_k(top_uris, relevantes, k):
    if not relevantes: return 0.0
    return len(set(top_uris[:k]) & relevantes) / len(relevantes)

def ndcg_at_k(top_uris, relevantes, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, u in enumerate(top_uris[:k]) if u in relevantes)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevantes), k)))
    return dcg / idcg if idcg > 0 else 0.0

def get_coocorrencias(uri_alvo, index_pl, uri_series, min_cooc=2):
    pls = index_pl.get(uri_alvo, set())
    if not pls: return set()
    return {u for u in uri_series if u != uri_alvo
            and len(pls & index_pl.get(u, set())) >= min_cooc}

print("Métricas prontas.")


# =============================================================================
# 8. Feature Selection — Ablation Study
# =============================================================================
ALL_FEATURES  = ['audio', 'genero', 'artista', 'ano', 'era', 'pais']

K_VALUES      = [5, 10, 20]
N_AMOSTRAS    = 300     
MIN_PLAYLISTS = 50      
MIN_COOC      = 2       
OUTPUT_CSV    = 'feature_selection.csv'
SEED          = 42

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

max_k      = max(K_VALUES)
uri_series = df['uri']
resultados = []

for config in tqdm(combinacoes, desc="Ablation"):
    features_ativas = [f for f, v in config.items() if v == 1]
    peso_unitario   = 1.0 / len(features_ativas)   

    acum   = {k: {'p': [], 'r': [], 'n': []} for k in K_VALUES}
    sem_gt = 0

    for _, row in amostra.iterrows():
        idx_alvo = row.name
        uri_alvo = row['uri']

        gt = get_coocorrencias(uri_alvo, track_playlists, uri_series, MIN_COOC)
        if not gt:
            sem_gt += 1
            continue

        score = np.zeros(len(df), dtype=np.float32)
        for feat in features_ativas:
            score += peso_unitario * SCORE_FUNCS[feat](idx_alvo)
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
df_sel.to_csv(OUTPUT_CSV, index=False)
print(f"\n{len(df_sel)} combinações avaliadas → {OUTPUT_CSV}")


# =============================================================================
# 9. Análise dos Resultados
# =============================================================================
df_sel = pd.read_csv(OUTPUT_CSV)

print(f"Total combinações testadas: {len(df_sel)}\n")

print("── Ranking por NDCG@10 ─────────────────────────────────────────────")
cols_show = ['features_ativas', 'n_features', 'ndcg@10', 'precision@10', 'recall@10', 'ndcg@5', 'ndcg@20']
print(df_sel.sort_values('ndcg@10', ascending=False)[cols_show].to_string(index=False))

print(f"\n{'Feature':<12}  {'ON (média)':>12}  {'OFF (média)':>12}  {'Delta':>10}  Veredicto")
print("-" * 68)

for feat in ALL_FEATURES:
    on  = df_sel[df_sel[feat] == 1]['ndcg@10']
    off = df_sel[df_sel[feat] == 0]['ndcg@10']
    delta = on.mean() - off.mean()
    if   delta >  0.002: veredicto = "AJUDA"
    elif delta < -0.002: veredicto = "ATRAPALHA"
    else:                veredicto = "➖ NEUTRO"
    print(f"{feat:<12}  {on.mean():>12.5f}  {off.mean():>12.5f}  {delta:>+10.5f}  {veredicto}")

print(f"\n{'N feat':>6}  {'NDCG@10':>8}  Features ativas")
print("-" * 55)
for n in sorted(df_sel['n_features'].unique()):
    sub  = df_sel[df_sel['n_features'] == n]
    best = sub.loc[sub['ndcg@10'].idxmax()]
    print(f"{n:>6}  {best['ndcg@10']:>8.5f}  {best['features_ativas']}")

# Aplicando a melhor combinação encontrada
best_row = df_sel.loc[df_sel['ndcg@10'].idxmax()]
features_best = best_row['features_ativas'].split('+')

print("\n" + "=" * 60)
print("MELHOR COMBINAÇÃO DE FEATURES")
print("=" * 60)
print(f"  Features ativas : {best_row['features_ativas']}")
print(f"  NDCG@10         : {best_row['ndcg@10']:.5f}")
print("=" * 60)

pesos_best = {f: 1.0 for f in features_best}

print()
r = recomendar("Billie Jean", pesos_best, top_k=20, excluir_mesmo_artista=True)
print(r.to_string(index=False))
