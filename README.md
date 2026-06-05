# Sistema de Recomendação de Músicas (TCC)

Comparação de abordagens de recomendação de músicas para playlists usando o
dataset **Spotify Million Playlist**. O projeto implementa três modelos sob uma
interface comum:

| Modelo | Arquivo | Status | Ideia |
|---|---|---|---|
| **Colaborativo** | [`model_collaborative.py`](model_collaborative.py) | ✅ Completo | Rede neural NeuMF (GMF + MLP) que aprende co-ocorrências playlist↔música |
| **Content-based** | [`model_content.py`](model_content.py) | ✅ Completo | TF-IDF (nome + artista) + similaridade de cosseno |
| **Híbrido** | [`model_hybrid.py`](model_hybrid.py) | 🚧 Parcial | Estrutura pronta; combina os dois sinais (colaborativo ainda a integrar) |

---

## 1. Pré-requisitos

- **Python 3.10**
- **(Opcional) GPU NVIDIA** com CUDA 12.x — acelera muito o treino do colaborativo
- Conexão com a internet na primeira execução (download do dataset via `kagglehub`)

> O `kagglehub` baixa o dataset automaticamente. Se for solicitado login, crie
> uma conta gratuita em [kaggle.com](https://www.kaggle.com) e gere um token em
> *Account → Create New API Token* (gera um `kaggle.json`).

## 2. Instalação

```bash
# (recomendado) criar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# instalar dependências
pip install -r requirements.txt
```

Para instalação **somente CPU** (sem GPU), edite o `requirements.txt` antes:
comente a linha `tensorflow[and-cuda]==2.21.0` e descomente `tensorflow==2.21.0`.

## 3. Como rodar

Toda a configuração fica no topo do [`main.py`](main.py):

```python
MODEL   = "collaborative"  # "collaborative" | "content" | "hybrid"
N_FILES = 300              # nº de arquivos JSON (cada arquivo = 1000 playlists)
```

Depois execute:

```bash
python main.py
```

- **Escolher o modelo:** altere `MODEL`.
- **Mais/menos dados:** altere `N_FILES` (ex.: `1000` = 1 milhão de playlists).
- **Conteúdo standalone:** `MODEL = "content"` ainda é um placeholder
  (lança `NotImplementedError`); use `python model_content.py` para rodá-lo.

### Rodar um módulo isoladamente

Cada arquivo também roda sozinho (útil para depurar):

```bash
python data_processing.py      # carrega e inspeciona os dados (não salva cache)
python model_content.py        # processa, salva o cache e avalia o content-based
python model_collaborative.py  # treina/carrega e avalia o colaborativo
```

> Na execução standalone, o `N_FILES` usado é o definido em
> [`data_processing.py`](data_processing.py), **não** o do `main.py`.

## 4. O que é gerado

| Pasta | Conteúdo | Quem cria |
|---|---|---|
| `cache/` | Dados pré-processados (TF-IDF, DataFrames, mapeamentos) | `model_content.save_cache()` |
| `saved_models/` | Rede colaborativa treinada (`collaborative_model.keras`) | `model_collaborative.save_model()` |

**Lógica de reuso:** se a pasta existe, o código **carrega** em vez de
reprocessar/retreinar. Para forçar do zero:

| Objetivo | Apague |
|---|---|
| Retreinar o colaborativo | `saved_models/collaborative_model.keras` |
| Reprocessar os dados | `cache/` |
| **Trocar `N_FILES`** | `cache/` **e** `saved_models/` |

> ⚠️ O `cache/` **não guarda** o valor de `N_FILES`. Se você mudar `N_FILES`
> sem apagar o `cache/`, o programa usará os dados antigos silenciosamente.

## 5. Como funciona a avaliação

Para cada playlist de teste, o sistema **remove uma fração** (`PCT_REMOVED`, 20%
por padrão) das músicas, pede ao modelo para recomendar, e mede quantas das
removidas reaparecem no **Top K** (métrica **Recall@K**). A semente (`SEED`)
fixa quais músicas são removidas, então diferentes modelos são comparados sobre
o **mesmo** conjunto — comparação justa.

Parâmetros de avaliação ficam no topo da seção correspondente de cada modelo
(ex.: `NUM_PLAYLISTS_TEST`, `PCT_REMOVED`, `TOP_K_REC` em
[`model_collaborative.py`](model_collaborative.py)).

## 6. Estrutura do projeto

```
.
├── main.py                  # ponto de entrada — seleciona e roda um modelo
├── data_processing.py       # download + pré-processamento (compartilhado)
├── model_collaborative.py   # modelo colaborativo (NeuMF) — baseline
├── model_content.py         # modelo content-based (TF-IDF)
├── model_hybrid.py          # modelo híbrido (estrutura)
├── requirements.txt         # dependências
├── cache/                   # (gerado) dados pré-processados
└── saved_models/            # (gerado) rede treinada
```
