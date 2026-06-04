# Нейронная сеть для анализа технологических трендов StackOverflow

Проект выполнен согласно индивидуальному заданию по производственной эксплуатационной практике.

## Тема

**Применение нейронных сетей для анализа технологических трендов StackOverflow**

## Что реализовано

- загрузка датасета `stackoverflow_combined.csv`;
- анализ структуры данных;
- очистка текста вопросов от HTML-разметки, ссылок и лишних символов;
- формирование текстового признака из `title`, `body`, `tags`;
- визуализация распределения языков программирования;
- анализ динамики технологий по годам;
- анализ `quality_score` и `difficulty_score`;
- векторизация текста методом TF-IDF;
- обучение baseline-модели Logistic Regression;
- обучение нескольких вариантов нейронной сети MLP;
- подбор параметров архитектуры, learning rate и регуляризации;
- расчет метрик Accuracy, Precision, Recall, F1-score;
- построение матрицы ошибок;
- сохранение лучшей модели.

## Структура проекта

```text
stackoverflow_nn_project/
│
├── src/
│   ├── main.py
│   └── predict.py
│
├── data/
│   └── stackoverflow_combined.csv
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── tables/
│
├── requirements.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

## Подготовка данных

Поместите файл датасета в папку:

```text
data/stackoverflow_combined.csv
```

## Запуск обучения

```bash
python src/main.py --data data/stackoverflow_combined.csv --target programming_language
```

Для обучения на всём датасете:

```bash
python src/main.py --data data/stackoverflow_combined.csv --target programming_language --sample-size 0
```

## Проверка модели на новом вопросе

```bash
python src/predict.py --model outputs/models/best_stackoverflow_model.joblib --text "How to create neural network in python?"
```

## Результаты

После запуска в папке `outputs` будут созданы:

- `figures/class_distribution.png` — распределение классов;
- `figures/technology_trend_by_year.png` — динамика технологий по годам;
- `figures/quality_score_by_class.png` — анализ качества вопросов;
- `figures/difficulty_score_by_class.png` — анализ сложности вопросов;
- `figures/confusion_matrix_best_model.png` — матрица ошибок;
- `tables/experiment_results.csv` — таблица экспериментов;
- `tables/best_model_report.txt` — отчет по лучшей модели;
- `models/best_stackoverflow_model.joblib` — сохраненная модель.

## Описание модели

Основная модель — многослойный перцептрон MLP.  
Входные данные представлены TF-IDF-векторами текста вопроса.  
Выходной слой классифицирует вопрос по технологическому направлению.

Используемые функции:

- ReLU — в скрытых слоях;
- Adam — алгоритм оптимизации;
- early stopping — остановка обучения при отсутствии улучшений;
- L2-регуляризация — снижение переобучения.

## Метрики

Для оценки качества используются:

- Accuracy;
- Precision macro;
- Recall macro;
- F1 macro.

Минимальное требование по заданию — accuracy не ниже 70%.
