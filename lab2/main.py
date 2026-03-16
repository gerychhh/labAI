from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT_DIR = Path(__file__).resolve().parents[1]
LAB_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset():
    train = pd.read_csv(ROOT_DIR / "processed_train.csv")
    valid = pd.read_csv(ROOT_DIR / "processed_valid.csv")
    data = pd.concat([train, valid], ignore_index=True)

    bool_columns = data.select_dtypes(include="bool").columns
    if len(bool_columns) > 0:
        data[bool_columns] = data[bool_columns].astype(int)

    return data


def regression_metrics(y_true, y_pred):
    return {
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def save_regression_plot(actual, linear_pred, poly_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    min_value = min(actual.min(), min(linear_pred), min(poly_pred))
    max_value = max(actual.max(), max(linear_pred), max(poly_pred))

    plots = [
        ("Линейная регрессия", linear_pred),
        ("Полиномиальная регрессия", poly_pred),
    ]

    for axis, (title, predicted) in zip(axes, plots):
        axis.scatter(actual, predicted, alpha=0.35, s=18, color="#1f4e79")
        axis.plot([min_value, max_value], [min_value, max_value], "--", color="#c0392b", linewidth=2)
        axis.set_title(title)
        axis.set_xlabel("Реальное значение")
        axis.set_ylabel("Предсказание")

    fig.suptitle("Регрессия для признака FoodCourt")
    fig.tight_layout()
    fig.savefig(LAB_DIR / "regression_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(matrix):
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=axis,
    )
    axis.set_title("Матрица ошибок")
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    fig.tight_layout()
    fig.savefig(LAB_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_dataset()

    print("Всего объектов:", len(data))
    print("Всего признаков:", len(data.columns))

    # 1. Разделение данных и решение задачи регрессии
    x_reg = data.drop(columns=["FoodCourt", "Transported"])
    y_reg = data["FoodCourt"]

    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
        x_reg,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    linear_reg = LinearRegression()
    linear_reg.fit(x_reg_train, y_reg_train)
    linear_pred = linear_reg.predict(x_reg_test)

    poly_reg = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear", LinearRegression()),
        ]
    )
    poly_reg.fit(x_reg_train, y_reg_train)
    poly_pred = poly_reg.predict(x_reg_test)

    linear_reg_metrics = regression_metrics(y_reg_test, linear_pred)
    poly_reg_metrics = regression_metrics(y_reg_test, poly_pred)
    save_regression_plot(y_reg_test, linear_pred, poly_pred)

    # 2. Разделение данных и решение задачи классификации
    x_cls = data.drop(columns=["Transported"])
    y_cls = data["Transported"].astype(int)

    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
        x_cls,
        y_cls,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    base_clf = LogisticRegression(max_iter=3000)
    base_clf.fit(x_cls_train, y_cls_train)
    base_pred = base_clf.predict(x_cls_test)

    improved_clf = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("logreg", LogisticRegression(max_iter=5000)),
        ]
    )
    improved_clf.fit(x_cls_train, y_cls_train)
    improved_pred = improved_clf.predict(x_cls_test)

    base_cls_metrics = {
        "Accuracy": round(float(accuracy_score(y_cls_test, base_pred)), 4),
        "Precision": round(float(precision_score(y_cls_test, base_pred)), 4),
        "Recall": round(float(recall_score(y_cls_test, base_pred)), 4),
        "F1": round(float(f1_score(y_cls_test, base_pred)), 4),
    }
    improved_cls_metrics = {
        "Accuracy": round(float(accuracy_score(y_cls_test, improved_pred)), 4),
        "Precision": round(float(precision_score(y_cls_test, improved_pred)), 4),
        "Recall": round(float(recall_score(y_cls_test, improved_pred)), 4),
        "F1": round(float(f1_score(y_cls_test, improved_pred)), 4),
    }

    matrix = confusion_matrix(y_cls_test, improved_pred).tolist()
    save_confusion_matrix(matrix)

    print("\nРегрессия: FoodCourt")
    print("Линейная регрессия:", linear_reg_metrics)
    print("Полиномиальная регрессия:", poly_reg_metrics)

    print("\nКлассификация: Transported")
    print("Базовая логистическая регрессия:", base_cls_metrics)
    print("Улучшенная логистическая регрессия:", improved_cls_metrics)
    print("Матрица ошибок:", matrix)

    print("\nСохранены файлы:")
    print(LAB_DIR / "regression_plot.png")
    print(LAB_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
