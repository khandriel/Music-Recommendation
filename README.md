# Sistema de Recomendação de Músicas (TCC)

Estudo comparativo de abordagens de recomendação para **continuação de playlists**
(*playlist continuation*) sobre o dataset **Spotify Million Playlist**. Dada uma
playlist com parte das faixas, cada modelo tenta prever as faixas removidas.

**Autores:** Arthur Real Sanchotene Ferreira e Osmar Sadi Nether Filho
**Orientador:** Luan Garcia

---

## Modelos comparados

Todos expõem a mesma interface (`load_or_train(...)` + `recommend(pid, seed_idxs, exclude_idxs, top_k)`):

| Modelo | Arquivo | Ideia |
|---|---|---|
| **Item-kNN** | [`model_collaborative_itemknn.py`](model_collaborative_itemknn.py) | Similaridade item-item por co-ocorrência em playlists (cosseno podado nos K vizinhos) |
| **ALS** | [`model_collaborative_als.py`](model_collaborative_als.py) | Matrix Factorization implícita (fatores latentes), com *fold-in* na inferência |
| **NeuMF** | [`model_collaborative_neumf.py`](model_collaborative_neumf.py) | Rede neural (GMF + MLP) que aprende a relação playlist↔música |
| **Content-Based** | [`model_content.py`](model_content.py) | Conteúdo da faixa (áudio, gênero, ano, artista, país); perfil-centroide da playlist |
| **Híbrido (late fusion)** | [`model_hybrid.py`](model_hybrid.py) | Combina os scores normalizados de um colaborativo + content (peso `ALPHA`) |

Os três colaborativos são arquivos **autocontidos** (cada um só treina + recomenda);
toda a avaliação fica centralizada no [`compare_models.py`](compare_models.py).

## Resultados de referência

Execução com `N_FILES=350` (350 mil playlists, ~1,34M músicas), 500 playlists de
teste, 20% das faixas removidas, `Top-500`, `seed=42` — números completos em
`comparison_report.txt`:

| Modelo | Recall@500 (micro) | NDCG@500 |
|---|---:|---:|
| Item-kNN | **65,8%** | 0,402 |
| ALS | 51,9% | 0,261 |
| NeuMF | 29,6% | 0,108 |
| Híbrido NeuMF+Content (α=0,5) | 23,0% | 0,092 |
| Content-Based (pesos afinados) | ~16,6% | ~0,077 |

---

## 1. Pré-requisitos

- **Python 3.10** (ambiente testado: conda `tcc`).
- Internet na primeira execução (o `kagglehub` baixa o dataset automaticamente).
- **(Opcional) GPU NVIDIA** para acelerar o NeuMF — ver seção GPU.
- Para o **Content-Based**, dois arquivos na raiz do projeto:
  - `audio_features.csv` — features de áudio por faixa (grande, **não versionado**).
  - `generos.json` — mapa `{artista: [gêneros]}` (já incluído no repositório).

> O `kagglehub` baixa o dataset sozinho. Se for pedido login, crie uma conta
> gratuita em [kaggle.com](https://www.kaggle.com) e gere um token em
> *Account → Create New API Token*.

## 2. Instalação

```bash
# (recomendado) criar um ambiente isolado
conda create -n tcc python=3.10
conda activate tcc

# instalar dependências
pip install -r requirements.txt
```

> **numpy < 2 é obrigatório:** o `tensorflow==2.10.0` não funciona com numpy 2.x
> (o `requirements.txt` já fixa `numpy==1.26.4`).

### GPU (opcional)

O TensorFlow 2.10 é a última versão com suporte nativo a GPU no Windows. Para
usar a GPU, instale à parte **CUDA 11.2 + cuDNN 8.1** (não vêm via pip nesta
versão). Sem isso, o NeuMF treina/infere em **CPU** normalmente (apenas mais lento).

## 3. Como rodar

O fluxo tem duas etapas: **treinar** os modelos e depois **comparar**.

### 3.1. Treinar — [`main.py`](main.py)

Escolha o que treinar nas flags do topo do arquivo e rode. Os modelos rodam
**um de cada vez** (sequencial — pensado para máquinas com RAM limitada); um
modelo já salvo em `saved_models/` é apenas carregado, não retreinado.

```python
TRAIN_ALS     = False   # Matrix Factorization implícita (leve)
TRAIN_ITEMKNN = False   # Item-item kNN por co-ocorrência (leve)
TRAIN_CONTENT = False   # Content-Based (a partir do audio_features.csv)
TRAIN_NEUMF   = False   # Rede neural NeuMF (PESADO — fica por último)
TRAIN_HYBRID  = False   # Híbrido NeuMF+Content (compõe; treina o que faltar)

N_FILES = 350           # arquivos JSON do dataset (cada um = 1000 playlists)
```

```bash
python main.py
```

### 3.2. Comparar — [`compare_models.py`](compare_models.py)

Avalia **todos** os modelos disponíveis em `saved_models/` sobre o mesmo
conjunto de teste e grava o relatório. Não treina nada (modelos ausentes são
pulados).

```bash
python compare_models.py        # gera comparison_report.txt
```

Parâmetros no topo do arquivo: `N_FILES`, `NUM_PLAYLISTS_TEST` (500),
`PCT_REMOVED` (0,20), `TOP_K` (500), `SEED` (42), `SHOW_DETAILS`.

### 3.3. Rodar um modelo isolado

Cada arquivo de modelo também roda sozinho (treina/carrega):

```bash
python model_collaborative_itemknn.py
python model_content.py
```

## 4. Ferramentas de análise (opcionais)

Scripts de medição — não fazem parte do pipeline, só imprimem tabelas no console
(não escrevem arquivos nem alteram modelos):

- [`sweep_hybrid_alpha.py`](sweep_hybrid_alpha.py) — varre o peso `ALPHA` do
  híbrido (0,0 → 1,0) e mostra Recall/NDCG em cada valor.
- [`sweep_pesos_content.py`](sweep_pesos_content.py) — varre os pesos das
  features do content (artista/gênero/ano) para afinar o `PESOS_DEFAULT`.

## 5. O que é gerado

| Caminho | Conteúdo | Quem cria |
|---|---|---|
| `cache/interactions_cache.joblib` | Interações pré-processadas + mapeamentos | `data_processing.py` |
| `saved_models/collaborative_als.npz` | ALS treinado | `main.py` (ou o próprio modelo) |
| `saved_models/collaborative_itemknn.npz` | Item-kNN treinado | idem |
| `saved_models/collaborative_neumf.keras` | NeuMF treinado | idem |
| `saved_models/content_based.joblib` | Catálogo do content (áudio/gênero/ano) | idem |
| `comparison_report.txt` | Relatório da comparação | `compare_models.py` |

> O híbrido **não** tem arquivo próprio — ele compõe o NeuMF e o content já salvos.

### Forçar reprocessamento/retreino

| Objetivo | Apague |
|---|---|
| Retreinar um modelo | o arquivo correspondente em `saved_models/` |
| Reprocessar os dados | `cache/` |

> O cache registra o `N_FILES` usado: se você mudar `N_FILES`, ele é invalidado
> e os dados são reprocessados **automaticamente** (não precisa apagar à mão).

## 6. Como funciona a avaliação

Para cada playlist de teste, remove-se `PCT_REMOVED` (20%) das faixas, pede-se o
`Top-K` ao modelo a partir das faixas restantes e mede-se quantas das removidas
reaparecem. A `SEED` fixa quais playlists entram e quais faixas são removidas —
**o mesmo conjunto para todos os modelos** (comparação justa). Métricas:
Recall@K (micro e macro), Precision@K, F1@K e NDCG@K.

## 7. Estrutura do projeto

```
.
├── main.py                          # treino (seleção por flags, sequencial)
├── compare_models.py                # avaliação + relatório (todos os modelos)
├── data_processing.py               # download + pré-processamento + cache
├── model_collaborative_neumf.py     # NeuMF (rede neural)
├── model_collaborative_als.py       # ALS (matrix factorization)
├── model_collaborative_itemknn.py   # item-item kNN
├── model_content.py                 # content-based (áudio/gênero/ano/artista)
├── model_hybrid.py                  # híbrido (late fusion)
├── sweep_hybrid_alpha.py            # análise: varredura do ALPHA do híbrido
├── sweep_pesos_content.py           # análise: varredura dos pesos do content
├── generos.json                     # {artista: [gêneros]} (para o content)
├── requirements.txt                 # dependências
├── cache/                           # (gerado) dados pré-processados
└── saved_models/                    # (gerado) modelos treinados
```
