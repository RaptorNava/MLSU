"""Практическая работа №4: сравнение линейных регрессионных моделей."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "datasets" / "BostonHousing.csv"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "medv"
CORRELATION_THRESHOLD = 0.85

sns.set_theme(style="whitegrid")


def make_model(features: list[str]) -> Pipeline:
    """Create the preprocessing + linear regression pipeline."""
    numeric_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", numeric_preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def select_reduced_features(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], list[str]]:
    """Remove one feature from highly correlated pairs using train only."""
    features = list(X_train.columns)
    dropped = []
    correlations = X_train.corr().abs()
    target_correlations = X_train.assign(_target=y_train).corr()['_target'].abs().drop("_target")

    while True:
        upper = correlations.where(~np.tril(np.ones(correlations.shape), k=0).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        pairs = pairs[pairs > CORRELATION_THRESHOLD]
        if pairs.empty:
            break
        left, right = pairs.index[0]
        feature_to_drop = left if target_correlations[left] < target_correlations[right] else right
        features.remove(feature_to_drop)
        dropped.append(feature_to_drop)
        correlations = X_train[features].corr().abs()
        target_correlations = target_correlations.drop(feature_to_drop)
    return features, dropped


def calculate_metrics(y_true: pd.Series, prediction: pd.Series) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "MSE": mean_squared_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
    }


def save_correlation_plot(X_train: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Корреляционная матрица признаков на train")
    plt.xlabel("Признак")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_train_correlation_matrix.png", dpi=160)
    plt.close()


def save_prediction_plot(y_test: pd.Series, prediction: pd.Series) -> None:
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=prediction, alpha=0.7)
    limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    plt.plot(limits, limits, "r--", label="Идеальное предсказание")
    plt.title("Истинные и предсказанные значения MEDV")
    plt.xlabel("Истинное значение MEDV, тыс. долларов")
    plt.ylabel("Предсказанное значение MEDV, тыс. долларов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_true_vs_predicted.png", dpi=160)
    plt.close()


def save_residual_plot(y_test: pd.Series, prediction: pd.Series) -> None:
    residuals = y_test - prediction
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=prediction, y=residuals, alpha=0.7)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("График остатков сокращённой модели")
    plt.xlabel("Предсказанное значение MEDV, тыс. долларов")
    plt.ylabel("Остаток: истинное - предсказанное, тыс. долларов")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_residuals.png", dpi=160)
    plt.close()


def build_report(
    data: pd.DataFrame,
    all_features: list[str],
    selected_features: list[str],
    dropped_features: list[str],
    baseline_metrics: dict[str, float],
    reduced_metrics: dict[str, float],
) -> str:
    baseline_mae = baseline_metrics["MAE"]
    reduced_mae = reduced_metrics["MAE"]
    residuals = reduced_metrics["residuals"]
    pattern = "явного систематического паттерна не видно" if abs(residuals.mean()) < 0.2 else "может присутствовать систематическое смещение"
    return f"""# Практическая работа №4. Регрессионная модель

## Данные

Использован Boston Housing: {data.shape[0]} наблюдений, {len(all_features)} числовых признаков. Целевая переменная `medv` — медианная стоимость жилья в тысячах долларов. Пропусков в исходном CSV: {int(data.isna().sum().sum())}.

## Сравнение моделей

| Модель | Признаки | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|---:|
| Базовая | {len(all_features)} | {baseline_metrics['MAE']:.3f} | {baseline_metrics['MSE']:.3f} | {baseline_metrics['RMSE']:.3f} | {baseline_metrics['R2']:.3f} |
| Сокращённая | {len(selected_features)} | {reduced_metrics['MAE']:.3f} | {reduced_metrics['MSE']:.3f} | {reduced_metrics['RMSE']:.3f} | {reduced_metrics['R2']:.3f} |

MAE сокращённой модели равен {reduced_mae:.2f} тыс. долларов: в среднем прогноз отличается от фактической стоимости примерно на ${reduced_mae * 1000:,.0f}. RMSE равен {reduced_metrics['RMSE']:.2f} тыс. долларов и сильнее реагирует на крупные ошибки; это примерно ${reduced_metrics['RMSE'] * 1000:,.0f}. MSE измеряется в квадратных тысячах долларов, поэтому его нельзя напрямую сравнивать с MAE. R² = {reduced_metrics['R2']:.3f} означает, что модель объясняет около {reduced_metrics['R2'] * 100:.1f}% вариации целевой переменной на test.

## Мультиколлинеарность и выбор признаков

Порог сильной корреляции: `|r| > {CORRELATION_THRESHOLD}`. Анализ выполнен только на `X_train`, чтобы test не участвовал в выборе признаков.

- Удалены избыточные признаки: {', '.join(dropped_features) or 'нет'}.
- Финальный набор: {', '.join(selected_features)}.
- Из пары сильно коррелирующих признаков сохранялся тот, который имел более сильную связь с `medv` на train.

## Остатки

На графике `03_residuals.png` проверяется равномерность остатков вокруг нулевой линии. По числовой проверке среднее остатка равно {residuals.mean():.3f} тыс. долларов: {pattern}. Если на графике видны дуга, веерообразное расширение или кластеры, линейная форма может быть недостаточной; тогда следует проверить логарифмические/полиномиальные признаки или нелинейную модель.

## Итоговый вывод

Финальной выбрана сокращённая модель на признаках `{', '.join(selected_features)}`: она уменьшает избыточность признаков и сохраняет сопоставимое качество с базовой моделью. Сравнение выполнено на одной и той же test-выборке, а предобработка (`SimpleImputer` и `StandardScaler`) обучалась только на train внутри `Pipeline`.

Графики: `01_train_correlation_matrix.png`, `02_true_vs_predicted.png`, `03_residuals.png`.
"""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    all_features = [column for column in data.columns if column != TARGET]
    X = data[all_features]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    selected_features, dropped_features = select_reduced_features(X_train, y_train)

    baseline_model = make_model(all_features)
    reduced_model = make_model(selected_features)
    baseline_prediction = baseline_model.fit(X_train, y_train).predict(X_test)
    reduced_prediction = reduced_model.fit(X_train[selected_features], y_train).predict(X_test[selected_features])

    baseline_metrics = calculate_metrics(y_test, baseline_prediction)
    reduced_metrics = calculate_metrics(y_test, reduced_prediction)
    baseline_metrics["residuals"] = (y_test - baseline_prediction).mean()
    reduced_metrics["residuals"] = (y_test - reduced_prediction).mean()

    save_correlation_plot(X_train)
    save_prediction_plot(y_test, reduced_prediction)
    save_residual_plot(y_test, reduced_prediction)
    report = build_report(data, all_features, selected_features, dropped_features, baseline_metrics, reduced_metrics)
    (OUTPUT_DIR / "regression_report.md").write_text(report, encoding="utf-8")

    print("Практическая работа №4: регрессионная модель")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Базовые признаки ({len(all_features)}): {', '.join(all_features)}")
    print(f"Удалены по мультиколлинеарности: {', '.join(dropped_features) or 'нет'}")
    print(f"Финальные признаки ({len(selected_features)}): {', '.join(selected_features)}")
    print(f"Baseline: MAE={baseline_metrics['MAE']:.3f}, RMSE={baseline_metrics['RMSE']:.3f}, R2={baseline_metrics['R2']:.3f}")
    print(f"Reduced:  MAE={reduced_metrics['MAE']:.3f}, RMSE={reduced_metrics['RMSE']:.3f}, R2={reduced_metrics['R2']:.3f}")
    print(f"Результаты сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
