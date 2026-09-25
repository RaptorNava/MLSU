"""CW3: leakage-safe preprocessing pipeline for the Titanic dataset."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "Class" / "data" / "titanic.csv"
TARGET = "Survived"


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features without calculating train/test statistics."""
    result = data.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    result["CabinDeck"] = result["Cabin"].fillna("Unknown").astype(str).str[0]
    result.loc[result["CabinDeck"].isin(["", "n"]), "CabinDeck"] = "Unknown"
    return result


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_features = ["Pclass", "Sex", "Embarked", "CabinDeck"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = pd.read_csv(DATA_PATH)
    data = add_features(data)

    X = data.drop(columns=TARGET)
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # fit выполняется только на X_train; X_test только преобразуется после обучения.
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    print("CW3: пайплайн предобработки Titanic")
    print(f"Исходный датасет: {data.shape[0]} строк, {data.shape[1]} столбцов после feature engineering")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    print(f"После преобразования: X_train={X_train_transformed.shape}, X_test={X_test_transformed.shape}")
    print(f"Размерности совпадают: {X_train_transformed.shape[1] == X_test_transformed.shape[1]}")
    print(f"Пропуски после преобразования: train={pd.isna(X_train_transformed).sum()}, test={pd.isna(X_test_transformed).sum()}")
    print(f"Количество итоговых признаков: {len(feature_names)}")
    print("Первые итоговые признаки:")
    print(", ".join(feature_names[:15]))
    print("\nПроверка утечки: статистики imputer/scaler вычислены только на train; test обработан через transform().")


if __name__ == "__main__":
    main()
