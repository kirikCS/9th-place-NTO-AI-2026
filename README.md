# НТО ИИ Финал — Кейс "Потеряшки", команда ICEQ

## Описание решения

Солюшен состоит из четырёх независимых пайплайнов для восстановления потерянных взаимодействий. Каждый пайплайн генерирует свой файл рекомендаций, финальный результат — RRF-бленд всех четырёх.

### Архитектура

**Pipeline 1 — LTR 1** (pipeline_ltr_v3.py → sub_ltr_v3.csv)
- Кандидаты: EASE^R, ALS, SVD, ItemKNN, Author continuation
- ~85 признаков (скоры моделей, SBERT-эмбеддинги, echo-сигналы, статистики)
- CatBoost YetiRank, graded labels (0/1/2), 4 фолда

**Pipeline 2 — LTR 2** (pipeline_v13.py → sub_v13.csv)
- Кандидаты: EASE@400, ALS@400, SVD@400, KNN, Author, Popular, Echo
- ~161 признак (domain ALS, text embeddings, z-scores, ранк-перцентили)
- CatBoost YetiRank, binary labels, fold decay

**Pipeline 3 — CF ансамбль** (pipeline_cf.py → sub_cf.csv)
- Item2ItemCF (count + cosine), UserUserCF, ALS (BM25), Author popularity
- Optuna-оптимизация весов и K для RRF бленда
- Валидация на скрытых 20% октябрьских позитивов

**Pipeline 4 — LTR 3** (pipeline_lgb_v4.py → sub_lgb_v4.csv)
- Кандидаты: U2U CF, I2I co-occurrence, Surge detection, Bridge patterns, Continuation, Author, EASE (enriched + incident), SVD
- ~90 признаков (CF-скоры, temporal, tabular, cross-features)
- LightGBM LambdaRank, 3 фолда, early stopping

**Финальный ансамбль** (main.py): RRF (K=60) по четырём CSV submission.csv

## Подготовка данных

Решение ожидает две директории с данными рядом со скриптами:

### 1. Основные данные — папка data/

Положите в неё датасет соревнования:
- interactions.csv
- editions.csv
- users.csv
- targets.csv
- book_genres.csv

### 2. Данные прошлого хакатона (обязательно)

Данные из прошлого хакатона критически важны для качества — они используются как дополнительный источник взаимодействий для EASE, KNN и других моделей.

Нужно подготовить **три варианта размещения** одних и тех же данных (разные пайплайны ищут их в разных местах):

**а)** Папка data_enriched/ — для пайплайна 1 (v3):
- interactions.csv — объединение текущих и прошлых взаимодействий
- book_genres.csv

**б)** Папка data-old-hackathon/ — для пайплайна 2 (v13):
- interactions.csv — взаимодействия из прошлого хакатона

**в)** Файл data/hack_interactions.csv — для пайплайна 4 (lgb_v4):
- Это те же взаимодействия прошлого хакатона, просто скопированные в data/ под именем hack_interactions.csv

### Итоговая структура


submission_bundle/
├── main.py
├── setup_validation.py
├── pipeline_ltr_v3.py
├── pipeline_v13.py
├── pipeline_cf.py
├── pipeline_lgb_v4.py
├── requirements.txt
├── README.md
├── AI_usage.md
├── data/
│   ├── interactions.csv
│   ├── editions.csv
│   ├── users.csv
│   ├── targets.csv
│   ├── book_genres.csv
│   └── hack_interactions.csv       ← копия interactions из прошлого хака
├── data_enriched/                   ← обогащённые данные
│   ├── interactions.csv             ← текущие + прошлые взаимодействия
│   └── book_genres.csv
└── data-old-hackathon/              ← данные прошлого хакатона
    └── interactions.csv


## Запуск

bash
pip install -r requirements.txt
python main.py


main.py — единственная точка входа. Он делает следующее:
1. Запускает setup_validation.py — создаёт валидационные сплиты (parquet-файлы) для Pipeline 3
2. Последовательно запускает все 4 пайплайна
3. Собирает результаты через RRF (K=60) в submission.csv

Если какой-то пайплайн упал (например, нет GPU или нет данных) — main.py пропустит его и соберёт бленд из оставшихся.

## Окружение и зависимости

- **Python**: 3.10+
- **GPU**: рекомендуется (CUDA) для CatBoost (Pipelines 1, 2) и implicit ALS
- **Без GPU**: Pipeline 3 (CF) и Pipeline 4 (LightGBM) работают на CPU

### requirements.txt


pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
catboost>=1.2
implicit>=0.7
sentence-transformers>=2.2
torch>=2.0
optuna>=3.0
tqdm>=4.60
lightgbm>=4.0


## Воспроизводимость

- random_seed=42 во всех пайплайнах (np.random.seed(42), CatBoost seed, LightGBM seed)
- Фиксированные гиперпараметры — нет случайного поиска в финальном запуске
- Фолды для валидации строятся детерминировано (simulate_hidden с фиксированным seed)
- Ожидаемое время: ~2–3 часа (с GPU), ~4–6 часов (CPU only)
