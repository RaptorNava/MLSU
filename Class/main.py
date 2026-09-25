"""Reproducible exploratory data analysis for the Titanic dataset."""

from pathlib import Path
import sys
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path(__file__).resolve().parent / "data" / "titanic.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_outputs"
TARGET = "Survived"

sns.set_theme(style="whitegrid", palette="Set2")


def classify_features(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns by meaning, keeping discrete numeric codes categorical."""
    numeric = ["Age", "Fare"]
    categorical = ["Sex", "Embarked", "Pclass", "SibSp", "Parch", TARGET]
    text = ["Name", "Ticket", "Cabin"]
    identifier = ["PassengerId"]
    known = set(numeric + categorical + text + identifier)

    for column in df.columns:
        if column not in known:
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric.append(column)
            else:
                categorical.append(column)
    return {
        "numeric": [column for column in numeric if column in df.columns],
        "categorical": [column for column in categorical if column in df.columns],
        "text": [column for column in text if column in df.columns],
        "identifier": [column for column in identifier if column in df.columns],
    }


def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    clean = series.dropna()
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum()), float(lower), float(upper)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def make_numeric_plots(df: pd.DataFrame, numeric: list[str]) -> list[str]:
    conclusions = []
    if not numeric:
        return conclusions

    rows = int(np.ceil(len(numeric) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, column in zip(axes, numeric):
        sns.histplot(df[column].dropna(), kde=True, ax=axis, color="#3b82f6")
        axis.set_title(f"Распределение: {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Количество наблюдений")
        skew = df[column].skew()
        shape = "с правым хвостом" if skew > 0.5 else "с левым хвостом" if skew < -0.5 else "близко к симметричному"
        conclusions.append(f"{column}: распределение {shape} (асимметрия {skew:.2f}).")
    for axis in axes[len(numeric):]:
        axis.remove()
    fig.suptitle("Гистограммы числовых признаков", y=1.02, fontsize=15)
    savefig("01_numeric_histograms.png")

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df[numeric], orient="h")
    plt.title("Boxplot числовых признаков")
    plt.xlabel("Значение признака")
    plt.ylabel("Признак")
    savefig("02_numeric_boxplots.png")
    outlier_notes = []
    for column in numeric:
        count, lower, upper = iqr_outliers(df[column])
        outlier_notes.append(f"{column}: {count} выбросов по правилу 1.5*IQR")
    conclusions.append("Boxplot: " + "; ".join(outlier_notes) + ".")
    return conclusions


def make_categorical_plots(df: pd.DataFrame, categorical: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    conclusions, rare = [], {}
    plot_columns = [column for column in categorical if column != TARGET]
    if not plot_columns:
        return conclusions, rare

    rows = int(np.ceil(len(plot_columns) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, column in zip(axes, plot_columns):
        counts = df[column].fillna("Пропуск").astype(str).value_counts().head(12)
        sns.barplot(x=counts.values, y=counts.index, ax=axis, color="#14b8a6")
        axis.set_title(f"Частоты: {column}")
        axis.set_xlabel("Количество")
        axis.set_ylabel(column)
        threshold = max(5, int(np.ceil(len(df) * 0.01)))
        rare[column] = counts[counts <= threshold].index.tolist()
        if rare[column]:
            conclusions.append(f"{column}: редкие категории/значения по критерию count <= {threshold}: {', '.join(rare[column])}.")
        else:
            conclusions.append(f"{column}: редких категорий по критерию count <= {threshold} не обнаружено.")
    for axis in axes[len(plot_columns):]:
        axis.remove()
    fig.suptitle("Частоты категориальных и дискретных признаков", y=1.02, fontsize=15)
    savefig("03_categorical_frequencies.png")
    return conclusions, rare


def make_target_plot(df: pd.DataFrame) -> str:
    counts = df[TARGET].value_counts(dropna=False)
    percentages = counts / len(df) * 100
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#f59e0b")
    ax.set_title("Баланс целевой переменной Survived")
    ax.set_xlabel("Класс Survived")
    ax.set_ylabel("Количество пассажиров")
    for index, (count, percent) in enumerate(zip(counts.values, percentages.values)):
        ax.text(index, count + len(df) * 0.015, f"{count} ({percent:.1f}%)", ha="center")
    savefig("04_target_balance.png")
    minority_share = percentages.min()
    return f"Целевая переменная Survived несбалансирована: меньший класс занимает {minority_share:.1f}% наблюдений."


def make_correlation_plots(df: pd.DataFrame, numeric: list[str], target: pd.Series) -> tuple[str, list[str]]:
    if len(numeric) < 2:
        return "Недостаточно числовых признаков для корреляционной матрицы.", []

    corr = df[numeric].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Корреляционная матрица числовых признаков")
    plt.xlabel("Числовой признак")
    plt.ylabel("Числовой признак")
    savefig("05_correlation_matrix.png")

    pairs = []
    for left_index, left in enumerate(numeric):
        for right in numeric[left_index + 1 :]:
            pairs.append((abs(corr.loc[left, right]), left, right))
    pairs = sorted(pairs, reverse=True)[:3]
    pair_columns = list(dict.fromkeys([column for _, left, right in pairs for column in (left, right)]))
    pair_data = df[pair_columns].copy()
    pair_data[TARGET] = target.astype(str).values
    grid = sns.pairplot(pair_data, vars=pair_columns, hue=TARGET, corner=True, diag_kind="hist")
    grid.fig.suptitle("Диаграммы рассеяния наиболее коррелирующих пар", y=1.02)
    grid.savefig(OUTPUT_DIR / "06_strongest_scatter_pairs.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    descriptions = [f"{left} и {right}: r = {corr.loc[left, right]:.2f}" for _, left, right in pairs]
    return "Корреляция показывает силу линейной связи, но сама по себе не доказывает причинность.", descriptions


def build_report(df: pd.DataFrame, groups: dict[str, list[str]], rare: dict[str, list[str]], numeric_notes: list[str], target_note: str, correlation_note: str, pair_notes: list[str]) -> str:
    missing = df.isna().sum()
    missing_text = ", ".join(f"{column}: {count}" for column, count in missing[missing > 0].items()) or "пропусков нет"
    outlier_text = []
    for column in groups["numeric"]:
        count, lower, upper = iqr_outliers(df[column])
        outlier_text.append(f"{column}: {count} по правилу 1.5*IQR (границы {lower:.2f}..{upper:.2f})")
    outlier_text = "; ".join(outlier_text) or "числовые признаки отсутствуют"
    duplicate_count = int(df.duplicated().sum())
    rare_text = "; ".join(f"{column}: {', '.join(values)}" for column, values in rare.items() if values) or "редкие категории не выявлены"
    pair_text = ", ".join(pair_notes) or "сильных пар не найдено"
    classifications = [
        f"Числовые: {', '.join(groups['numeric']) or 'нет'}.",
        f"Категориальные/дискретные: {', '.join(groups['categorical']) or 'нет'}.",
        f"Текстовые: {', '.join(groups['text']) or 'нет'}; идентификаторы: {', '.join(groups['identifier']) or 'нет'}.",
    ]
    bullets = [
        f"Размер данных: {df.shape[0]} строк и {df.shape[1]} признаков. Классификация признаков: {' '.join(classifications)}",
        f"Пропуски: {missing_text}. На следующем занятии для Age планируется медианное заполнение с учетом пола и класса, для Embarked — мода; Cabin следует преобразовать в признак наличия/палубы или обоснованно исключить.",
        f"Дубликаты: {duplicate_count}. Повторы нужно проверить по PassengerId и бизнес-смыслу перед удалением, чтобы не потерять реальные записи.",
        f"Выбросы по правилу 1.5*IQR: {outlier_text}. Удалять их автоматически не следует: Fare имеет естественный длинный хвост, поэтому сначала нужна проверка влияния и, возможно, лог-преобразование.",
        f"Категориальные частоты: {rare_text}. Редкие уровни нужно объединить в 'Other' или закодировать устойчивым способом; Ticket и Name требуют выделения осмысленных признаков вместо прямого кодирования строк.",
        f"{target_note} На следующем занятии при обучении модели нужно использовать стратифицированное разбиение и метрики по каждому классу.",
        f"{correlation_note} Наиболее сильные пары: {pair_text}. Связи следует проверить после кодирования категорий и не трактовать как причинно-следственные.",
    ]
    graph_notes = numeric_notes + [target_note, correlation_note] + pair_notes
    return "ОТЧЕТ ПО EDA\n\n" + "\n".join(f"{index}. {bullet}" for index, bullet in enumerate(bullets, 1)) + "\n\nВЫВОДЫ ПО ГРАФИКАМ\n\n" + "\n".join(f"- {note}" for note in graph_notes)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    groups = classify_features(df)

    print(f"Датасет: {DATA_PATH}")
    print(f"Размер: {df.shape[0]} строк x {df.shape[1]} столбцов")
    print("\nТипы и пропуски:")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique(dropna=False)}).to_string())
    print("\nКлассификация признаков:")
    for group, columns in groups.items():
        print(f"{group}: {', '.join(columns) or 'нет'}")
    print("\nСтатистики числовых признаков:")
    print(df[groups["numeric"]].describe().T.to_string())
    print(f"\nДубликаты: {df.duplicated().sum()}")

    numeric_notes = make_numeric_plots(df, groups["numeric"])
    categorical_notes, rare = make_categorical_plots(df, groups["categorical"])
    target_note = make_target_plot(df)
    correlation_note, pair_notes = make_correlation_plots(df, groups["numeric"], df[TARGET])
    report = build_report(df, groups, rare, numeric_notes + categorical_notes, target_note, correlation_note, pair_notes)
    report_path = OUTPUT_DIR / "eda_report.txt"
    report_path.write_text(dedent(report), encoding="utf-8")
    print(f"\nОтчет сохранен: {report_path}")
    print(f"Графики сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
