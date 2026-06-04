
"""
Проект по производственной практике:
Применение нейронных сетей для анализа технологических трендов StackOverflow.

Скрипт выполняет:
1) загрузку и первичный анализ датасета;
2) очистку текстовых данных;
3) визуализацию распределений и трендов;
4) подготовку признаков TF-IDF;
5) обучение baseline-модели и нейронной сети MLP;
6) эксперименты с параметрами нейросети;
7) расчет метрик качества;
8) сохранение графиков, таблиц и обученной модели.

Запуск:
python src/main.py --data data/stackoverflow_combined.csv --target programming_language
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    params: Dict


def ensure_dirs(output_dir: Path) -> None:
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл датасета не найден: {path}")

    df = pd.read_csv(path)
    print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def clean_html(text: str) -> str:
    text = str(text)
    text = re.sub(r"<code>.*?</code>", " code_block ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9+#.\s_-]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df: pd.DataFrame, target: str, max_classes: int) -> pd.DataFrame:
    required = ["title", "body", "tags", target]
    missed = [col for col in required if col not in df.columns]
    if missed:
        raise ValueError(f"В датасете отсутствуют обязательные столбцы: {missed}")

    work = df.copy()
    work = work.dropna(subset=["title", "body", "tags", target])

    # Для устойчивого обучения оставляем наиболее представленные классы.
    top_classes = work[target].value_counts().head(max_classes).index
    work = work[work[target].isin(top_classes)].copy()

    work["text"] = (
        work["title"].fillna("").astype(str)
        + " "
        + work["body"].fillna("").astype(str)
        + " "
        + work["tags"].fillna("").astype(str)
    )
    work["clean_text"] = work["text"].apply(clean_html)

    # Удаляем слишком короткие тексты, так как они почти не несут смысловых признаков.
    work["clean_word_count"] = work["clean_text"].str.split().apply(len)
    work = work[work["clean_word_count"] >= 5].copy()

    print("После очистки:", work.shape)
    print("Классы:")
    print(work[target].value_counts())

    return work


def save_dataset_overview(df: pd.DataFrame, output_dir: Path) -> None:
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[col].dtype) for col in df.columns],
        "missing_values": [int(df[col].isna().sum()) for col in df.columns],
        "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
    })
    overview.to_csv(output_dir / "tables" / "dataset_overview.csv", index=False, encoding="utf-8-sig")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols].describe().T.to_csv(
            output_dir / "tables" / "numeric_statistics.csv",
            encoding="utf-8-sig"
        )


def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_visualizations(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    figures = output_dir / "figures"

    plot_bar(
        df[target].value_counts().head(15),
        "Распределение классов технологических направлений",
        "Класс",
        "Количество вопросов",
        figures / "class_distribution.png"
    )

    if "creation_year" in df.columns:
        trend = df.groupby(["creation_year", target]).size().unstack(fill_value=0)
        top_cols = df[target].value_counts().head(8).index
        trend = trend[top_cols]

        plt.figure(figsize=(11, 6))
        for col in trend.columns:
            plt.plot(trend.index, trend[col], marker="o", label=col)
        plt.title("Динамика популярности технологий по годам")
        plt.xlabel("Год")
        plt.ylabel("Количество вопросов")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "technology_trend_by_year.png", dpi=200)
        plt.close()

    if "quality_score" in df.columns:
        quality = df.groupby(target)["quality_score"].mean().sort_values(ascending=False).head(15)
        plot_bar(
            quality,
            "Средний quality_score по технологиям",
            "Технология",
            "Среднее значение quality_score",
            figures / "quality_score_by_class.png"
        )

    if "difficulty_score" in df.columns:
        difficulty = df.groupby(target)["difficulty_score"].mean().sort_values(ascending=False).head(15)
        plot_bar(
            difficulty,
            "Средний difficulty_score по технологиям",
            "Технология",
            "Среднее значение difficulty_score",
            figures / "difficulty_score_by_class.png"
        )


def evaluate_model(name: str, model: Pipeline, x_test: pd.Series, y_test: pd.Series, params: Dict) -> Tuple[ExperimentResult, np.ndarray, str]:
    y_pred = model.predict(x_test)

    result = ExperimentResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision_macro=float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        params=params,
    )

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    report = classification_report(y_test, y_pred, zero_division=0)

    return result, cm, report


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Матрица ошибок финальной нейронной сети")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_baseline(max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


def build_mlp(max_features: int, hidden_layer_sizes: Tuple[int, ...], learning_rate_init: float, alpha: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            stop_words=list(ENGLISH_STOP_WORDS),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=25,
            batch_size=256,
            # early_stopping отключен, так как в новых версиях scikit-learn
            # он может вызывать ошибку np.isnan для строковых меток классов.
            early_stopping=False,
            random_state=42,
            verbose=False
        ))
    ])


def run_experiments(df: pd.DataFrame, target: str, output_dir: Path, sample_size: int | None) -> None:
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).copy()
        print(f"Для ускорения обучения использована выборка: {sample_size} строк")

    x = df["clean_text"]
    y = df[target].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    experiments = []

    models = [
        (
            "baseline_logistic_regression",
            build_baseline(max_features=8000),
            {"model": "LogisticRegression", "max_features": 8000}
        ),
        (
            "mlp_small",
            build_mlp(max_features=5000, hidden_layer_sizes=(128,), learning_rate_init=0.001, alpha=0.0001),
            {"model": "MLP", "max_features": 5000, "hidden_layers": [128], "learning_rate": 0.001, "alpha": 0.0001}
        ),
        (
            "mlp_medium",
            build_mlp(max_features=8000, hidden_layer_sizes=(256, 128), learning_rate_init=0.001, alpha=0.0001),
            {"model": "MLP", "max_features": 8000, "hidden_layers": [256, 128], "learning_rate": 0.001, "alpha": 0.0001}
        ),
        (
            "mlp_regularized",
            build_mlp(max_features=8000, hidden_layer_sizes=(256, 128), learning_rate_init=0.0005, alpha=0.001),
            {"model": "MLP", "max_features": 8000, "hidden_layers": [256, 128], "learning_rate": 0.0005, "alpha": 0.001}
        ),
    ]

    best_model = None
    best_result = None
    best_cm = None
    best_report = ""

    for name, model, params in models:
        print(f"\nОбучение модели: {name}")
        model.fit(x_train, y_train)
        result, cm, report = evaluate_model(name, model, x_test, y_test, params)
        experiments.append(result.__dict__)

        print(f"Accuracy: {result.accuracy:.4f}; F1-macro: {result.f1_macro:.4f}")

        with open(output_dir / "tables" / f"{name}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        if best_result is None or result.f1_macro > best_result.f1_macro:
            best_model = model
            best_result = result
            best_cm = cm
            best_report = report

    results_df = pd.DataFrame(experiments)
    results_df.to_csv(output_dir / "tables" / "experiment_results.csv", index=False, encoding="utf-8-sig")

    labels = sorted(y_test.unique())
    plot_confusion_matrix(best_cm, labels, output_dir / "figures" / "confusion_matrix_best_model.png")

    with open(output_dir / "tables" / "best_model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Лучшая модель: {best_result.name}\n")
        f.write(json.dumps(best_result.__dict__, ensure_ascii=False, indent=4))
        f.write("\n\nClassification report:\n")
        f.write(best_report)

    joblib.dump(best_model, output_dir / "models" / "best_stackoverflow_model.joblib")

    print("\nЛучшая модель:")
    print(json.dumps(best_result.__dict__, ensure_ascii=False, indent=4))
    print(f"Модель сохранена: {output_dir / 'models' / 'best_stackoverflow_model.joblib'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Путь к CSV-файлу датасета")
    parser.add_argument("--target", type=str, default="programming_language", help="Целевая переменная")
    parser.add_argument("--output", type=str, default="outputs", help="Папка для результатов")
    parser.add_argument("--max-classes", type=int, default=8, help="Количество наиболее частых классов")
    parser.add_argument("--sample-size", type=int, default=30000, help="Размер выборки для ускорения обучения; 0 = весь датасет")
    args = parser.parse_args()

    output_dir = Path(args.output)
    ensure_dirs(output_dir)

    df = load_data(Path(args.data))
    save_dataset_overview(df, output_dir)

    prepared = preprocess_data(df, target=args.target, max_classes=args.max_classes)
    prepared[["question_id", "clean_text", args.target]].to_csv(
        output_dir / "tables" / "prepared_dataset_sample.csv",
        index=False,
        encoding="utf-8-sig"
    )

    make_visualizations(prepared, target=args.target, output_dir=output_dir)

    sample_size = None if args.sample_size == 0 else args.sample_size
    run_experiments(prepared, target=args.target, output_dir=output_dir, sample_size=sample_size)

    print("\nРабота завершена. Результаты сохранены в папке outputs.")


if __name__ == "__main__":
    main()
