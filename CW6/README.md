# Классификация тональности IMDB: TF-IDF + логистическая регрессия

Пайплайн предобработки текста и классификации отзывов (positive / negative).

**Данные:** IMDB, подвыборка 15 000 отзывов, классы сбалансированы. Два варианта:

- **А (по умолчанию):** [`stanfordnlp/imdb` на Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb), скачивается автоматически.
- **Б (если Hugging Face недоступен):** скачать [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) с Kaggle и положить `IMDB Dataset.csv` рядом со скриптом. Он подхватится сам.

## Запуск

```bash
pip install -r requirements.txt
python pipeline.py
```

Данные скачиваются автоматически. Параметры (размер выборки, `min_df`, `max_df`, `SEED`) заданы в начале `pipeline.py`.

## Что делает пайплайн

1. **Разбиение 80/20 со `stratify`** до любой обработки. Векторизатор обучается только на train, утечки нет.
2. **Очистка:** нижний регистр, удаление HTML, пунктуации и цифр, стоп-слов. Отрицания (`not`, `no`, `nor`, `very`, `too`, `but`) оставлены, `n't` раскрывается в `not`.
3. **Токенизация и стемминг** (Snowball). Для сравнения считается вариант без стемминга.
4. **TF-IDF:** `min_df=5`, `max_df=0.5`, n-граммы (1, 2). Выводится размерность матрицы и влияние `min_df` (1 / 5 / 20).
5. **Классификатор:** `LogisticRegression(class_weight="balanced")`, baseline: `DummyClassifier`.
6. **Метрики:** accuracy, macro-F1, precision/recall по классам, матрица ошибок.
7. **Анализ 5 ошибок:** 2 уверенных FP, 2 уверенных FN, 1 пограничная. Для каждой: текст, токены после очистки, вклад признаков, автоматические гипотезы.

## Результаты (папка `outputs/`)

| Файл | Содержимое |
|---|---|
| `confusion_matrix.png` | матрица ошибок на test |
| `errors_all.csv` | все неверно классифицированные тексты |
| `errors_report.md` | разбор 5 ошибок с гипотезами |

