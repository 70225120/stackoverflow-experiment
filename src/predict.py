
"""
Проверка обученной модели на новом вопросе StackOverflow.

Пример:
python src/predict.py --model outputs/models/best_stackoverflow_model.joblib --text "How to create dataframe in python pandas?"
"""

from __future__ import annotations

import argparse
import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Путь к файлу модели .joblib")
    parser.add_argument("--text", required=True, help="Текст вопроса")
    args = parser.parse_args()

    model = joblib.load(args.model)
    prediction = model.predict([args.text])[0]

    print(f"Предсказанный класс: {prediction}")

    if hasattr(model[-1], "predict_proba"):
        probabilities = model.predict_proba([args.text])[0]
        classes = model[-1].classes_
        top = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)[:5]
        print("Вероятности:")
        for cls, prob in top:
            print(f"{cls}: {prob:.4f}")


if __name__ == "__main__":
    main()
