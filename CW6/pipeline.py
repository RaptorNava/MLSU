"""
Пайплайн предобработки текста + TF-IDF + классификатор.
Датасет: IMDB (бинарная тональность отзывов), Hugging Face `stanfordnlp/imdb`
или CSV с Kaggle ("IMDB Dataset.csv" рядом со скриптом).

Запуск:  python pipeline.py
Результаты сохраняются в папку outputs/.
"""
import os
import re
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

# ----------------------------- Параметры -----------------------------
SEED = 42
N_SAMPLES = 15000          # подвыборка корпуса (для скорости)
TEST_SIZE = 0.2
MIN_DF = 5                 # слово должно встречаться минимум в 5 документах
MAX_DF = 0.5               # и не более чем в 50% документов
NGRAM_RANGE = (1, 2)       # униграммы + биграммы (сохраняет "not good")
OUT_DIR = "outputs"
LOCAL_CSV = "IMDB Dataset.csv"   # если файл лежит рядом, данные берутся из него
LABELS = {0: "negative", 1: "positive"}

os.makedirs(OUT_DIR, exist_ok=True)
nltk.download("stopwords", quiet=True)


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ------------------------- 1. Загрузка данных -------------------------
section("1. ДАННЫЕ")
if os.path.exists(LOCAL_CSV):
    # Вариант Б: CSV с Kaggle (столбцы review, sentiment)
    print(f"Читаю локальный файл {LOCAL_CSV}")
    df = pd.read_csv(LOCAL_CSV).rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = (df["label"] == "positive").astype(int)
else:
    # Вариант А: Hugging Face (скачивается автоматически)
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
df = df.sample(min(N_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)

print("Размер корпуса:", df.shape)
print("Доли классов:\n", df["label"].map(LABELS).value_counts(normalize=True).round(3))

# ---------------- 2. Разбиение ДО любой обработки/обучения ----------------
# Векторизатор будет обучаться только на train -> утечки данных нет.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=TEST_SIZE, random_state=SEED, stratify=df["label"],
)
print(f"\nTrain: {len(X_train_raw)}, Test: {len(X_test_raw)}")

# ------------------------- 3. Очистка текста -------------------------
section("2. ПРЕДОБРАБОТКА")

stop = set(stopwords.words("english"))
# Слова, несущие смысл для тональности, из стоп-слов убираем:
KEEP = {"no", "nor", "not", "very", "too", "against", "but"}
stop -= KEEP
print("Оставлены (не считаются стоп-словами):", sorted(KEEP))
print("Размер списка стоп-слов:", len(stop))

stemmer = SnowballStemmer("english")


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    return stemmer.stem(word)


def clean_text(text: str) -> str:
    """Нижний регистр, удаление HTML/пунктуации/цифр, стоп-слов."""
    text = text.lower()                                   # 1. нижний регистр
    text = re.sub(r"<[^>]+>", " ", text)                  # HTML-теги (<br />)
    text = re.sub(r"\bcan't\b", "can not", text)          # раскрываем сокращения,
    text = re.sub(r"\bwon't\b", "will not", text)         # чтобы не потерять "not"
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)                 # 2. пунктуация и цифры
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop]   # 3. стоп-слова (токенизация)
    return " ".join(tokens)


def stem_text(text: str) -> str:
    return " ".join(stem(w) for w in text.split())


def preprocess(text: str) -> str:
    return stem_text(clean_text(text))


print("\nПримеры до/после очистки:")
for t in X_train_raw.iloc[:3]:
    print("RAW  :", t[:200].replace("\n", " "))
    print("CLEAN:", clean_text(t)[:200])
    print("STEM :", preprocess(t)[:200], "\n")

test_sentence = "This movie wasn't good, I don't like it. 10/10 never again!"
print("Проверка отрицаний:")
print("  вход :", test_sentence)
print("  выход:", clean_text(test_sentence))

# Очищенный текст без стемминга и со стеммингом
X_train_clean = X_train_raw.apply(clean_text)
X_test_clean = X_test_raw.apply(clean_text)
X_train = X_train_clean.apply(stem_text)
X_test = X_test_clean.apply(stem_text)

# --------------------------- 4. TF-IDF ---------------------------
section("3. ВЕКТОРИЗАЦИЯ (TF-IDF)")


def make_vec(min_df=MIN_DF, max_df=MAX_DF) -> TfidfVectorizer:
    return TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=NGRAM_RANGE)


vec = make_vec()
X_train_vec = vec.fit_transform(X_train)     # fit ТОЛЬКО на train
X_test_vec = vec.transform(X_test)           # на test только transform

print(f"Параметры: min_df={MIN_DF}, max_df={MAX_DF}, ngram_range={NGRAM_RANGE}")
print("Размерность train:", X_train_vec.shape, "(документы x признаки)")
print("Размерность test :", X_test_vec.shape)
print("Размер словаря   :", len(vec.vocabulary_))

print("\nВлияние min_df на размер словаря (fit на train):")
for md in [1, 5, 20]:
    n = make_vec(min_df=md).fit(X_train).get_feature_names_out().shape[0]
    print(f"  min_df={md:>2} -> признаков: {n}")

# Сравнение: со стеммингом и без
vec_ns = make_vec()
Xtr_ns = vec_ns.fit_transform(X_train_clean)
Xte_ns = vec_ns.transform(X_test_clean)
print(f"\nБез стемминга: {Xtr_ns.shape[1]} признаков | со стеммингом: {X_train_vec.shape[1]}")

# --------------------------- 5. Модель ---------------------------
section("4. КЛАССИФИКАТОР И МЕТРИКИ")

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

clf_ns = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
clf_ns.fit(Xtr_ns, y_train)
f1_ns = f1_score(y_test, clf_ns.predict(Xte_ns), average="macro")

baseline = DummyClassifier(strategy="most_frequent").fit(X_train_vec, y_train)
y_base = baseline.predict(X_test_vec)

print(f"Baseline (most_frequent): accuracy={accuracy_score(y_test, y_base):.3f}, "
      f"macro-F1={f1_score(y_test, y_base, average='macro'):.3f}")
print(f"LogReg без стемминга    : macro-F1={f1_ns:.3f}")
print(f"LogReg со стеммингом    : accuracy={accuracy_score(y_test, y_pred):.3f}, "
      f"macro-F1={f1_score(y_test, y_pred, average='macro'):.3f}\n")
print(classification_report(y_test, y_pred, target_names=list(LABELS.values()), digits=3))

cm = confusion_matrix(y_test, y_pred)
print("Матрица ошибок (строки - истина, столбцы - предсказание):\n", cm)
ConfusionMatrixDisplay(cm, display_labels=list(LABELS.values())).plot(cmap="Blues")
plt.title("Confusion matrix (test)")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

feat = np.array(vec.get_feature_names_out())
coef = clf.coef_[0]
print("\nТоп-15 признаков 'positive':", list(feat[np.argsort(coef)[-15:][::-1]]))
print("Топ-15 признаков 'negative':", list(feat[np.argsort(coef)[:15]]))

# ------------------------ 6. Анализ ошибок ------------------------
section("5. АНАЛИЗ 5 ОШИБОК")

proba = clf.predict_proba(X_test_vec)
errors = pd.DataFrame({
    "idx": np.arange(len(y_test)),
    "text": X_test_raw.values,
    "clean": X_test.values,
    "true": y_test.values,
    "pred": y_pred,
    "conf": proba.max(axis=1),
})
errors = errors[errors["true"] != errors["pred"]].copy()
errors["type"] = np.where(errors["pred"] == 1, "FP (ждали neg, предсказан pos)",
                          "FN (ждали pos, предсказан neg)")
errors.drop(columns="idx").to_csv(f"{OUT_DIR}/errors_all.csv", index=False)
print(f"Всего ошибок: {len(errors)} из {len(y_test)} "
      f"(FP: {(errors['pred'] == 1).sum()}, FN: {(errors['pred'] == 0).sum()})")

# Выбор 5 ошибок: 2 самых уверенных FP, 2 самых уверенных FN, 1 самая неуверенная
fp = errors[errors["pred"] == 1].sort_values("conf", ascending=False).head(2)
fn = errors[errors["pred"] == 0].sort_values("conf", ascending=False).head(2)
rest = errors.drop(index=list(fp.index) + list(fn.index))
borderline = rest.sort_values("conf", ascending=True).head(1)
selected = pd.concat([fp, fn, borderline])

NEGATIONS = {"not", "no", "nor", "never", "nothing", "nobody", "neither", "without"}


def diagnose(row) -> dict:
    """Автодиагностика: вклад признаков, отрицания, длина, слова вне словаря."""
    x = X_test_vec[row["idx"]].toarray().ravel()
    contrib = x * coef                          # вклад в класс 'positive'
    order = np.argsort(contrib)
    pos_top = [(str(feat[i]), round(float(contrib[i]), 3)) for i in order[::-1][:5] if contrib[i] > 0]
    neg_top = [(str(feat[i]), round(float(contrib[i]), 3)) for i in order[:5] if contrib[i] < 0]
    tokens = row["clean"].split()
    unigrams = set(vec.vocabulary_)
    oov = np.mean([t not in unigrams for t in tokens]) if tokens else 0.0
    pos_sum, neg_sum = contrib[contrib > 0].sum(), -contrib[contrib < 0].sum()
    mixed = min(pos_sum, neg_sum) / max(pos_sum, neg_sum, 1e-9)
    return dict(pos_top=pos_top, neg_top=neg_top, n_tokens=len(tokens), oov=oov,
                has_neg=any(t in NEGATIONS for t in tokens), mixed=mixed)


def hypotheses(row, d) -> list:
    """Правила-кандидаты. Их нужно проверить глазами и подтвердить/опровергнуть."""
    h = []
    if d["mixed"] > 0.6:
        h.append("смешанная тональность: сильные позитивные и негативные признаки уравновешивают друг друга")
    if d["has_neg"]:
        h.append("отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'")
    if d["n_tokens"] < 40:
        h.append("короткий текст: мало признаков для уверенного решения")
    if d["n_tokens"] > 400:
        h.append("очень длинный текст: общий вывод 'тонет' среди описания сюжета")
    if d["oov"] > 0.3:
        h.append("много слов вне словаря (отсечены min_df/max_df или не было в train)")
    if row["conf"] > 0.85 and not h:
        h.append("уверенная ошибка без явных признаков: вероятны ирония/сарказм или шум в разметке")
    if not h:
        h.append("пограничный случай: модель сама не уверена, смысл выражен неявно")
    return h


lines = ["# Анализ 5 ошибок модели\n",
         "Гипотезы ниже сформированы автоматически по простым правилам. "
         "Прочитайте каждый текст и подтвердите/исправьте гипотезу вручную.\n"]
for n, (_, row) in enumerate(selected.iterrows(), 1):
    d = diagnose(row)
    hyp = hypotheses(row, d)
    snippet = row["text"].replace("<br />", " ").replace("\n", " ")[:500]
    block = [
        f"## Ошибка №{n} — {row['type']}",
        f"- **Истина / предсказание:** {LABELS[row['true']]} / {LABELS[row['pred']]} "
        f"(уверенность {row['conf']:.2f})",
        f"- **Текст:** «{snippet}...»",
        f"- **После очистки:** {row['clean'][:300]}...",
        f"- **Токенов после очистки:** {d['n_tokens']}, доля слов вне словаря: {d['oov']:.2f}",
        f"- **Топ вкладов в 'positive':** {d['pos_top']}",
        f"- **Топ вкладов в 'negative':** {d['neg_top']}",
        f"- **Гипотеза(ы):** " + "; ".join(hyp),
        "- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_\n",
    ]
    lines += block
    print("\n".join(block))

with open(f"{OUT_DIR}/errors_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

section("ГОТОВО")
print(f"Файлы сохранены в {OUT_DIR}/: confusion_matrix.png, errors_all.csv, errors_report.md")