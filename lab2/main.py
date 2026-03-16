from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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

LAB_DIR = Path(__file__).resolve().parent
DATASET_PATH = LAB_DIR / "dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset():
    data = pd.read_csv(DATASET_PATH)

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


def save_regression_plot(actual, linear_pred, boosted_pred, linear_metrics, boosted_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sorted_data = pd.DataFrame(
        {
            "actual": actual.reset_index(drop=True),
            "linear": linear_pred,
            "boosted": boosted_pred,
        }
    ).sort_values("actual", ignore_index=True)
    window = max(25, len(sorted_data) // 40)
    smooth = sorted_data.rolling(window=window, min_periods=1).mean()

    axes[0].scatter(
        sorted_data.index,
        sorted_data["actual"],
        s=8,
        alpha=0.12,
        color="#1f4e79",
        label="Реальные значения (точки)",
    )
    axes[0].plot(smooth["actual"], label="Реальные значения (тренд)", color="#1f4e79", linewidth=2.2)
    axes[0].plot(smooth["linear"], label="Линейная регрессия", color="#7f8c8d", linewidth=1.8)
    axes[0].plot(smooth["boosted"], label="Градиентный бустинг", color="#c0392b", linewidth=2)
    axes[0].set_title("Сглаженный тренд предсказаний")
    axes[0].set_xlabel("Объекты тестовой выборки\n(отсортированы по реальному значению)")
    axes[0].set_ylabel("Значение признака FoodCourt")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[0].text(
        0.02,
        0.02,
        f"Окно сглаживания: {window}",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#555555",
    )

    metric_names = ["RMSE", "MAE", "R2"]
    x = range(len(metric_names))
    width = 0.35
    axes[1].bar(
        [i - width / 2 for i in x],
        [linear_metrics[name] for name in metric_names],
        width=width,
        label="Линейная регрессия",
        color="#7f8c8d",
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        [boosted_metrics[name] for name in metric_names],
        width=width,
        label="Градиентный бустинг",
        color="#1f4e79",
    )
    axes[1].set_xticks(list(x), metric_names)
    axes[1].set_title("Сравнение метрик")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    for bars in axes[1].containers:
        axes[1].bar_label(bars, fmt="%.3f", fontsize=9, padding=3)

    fig.suptitle("Регрессия для признака FoodCourt", fontsize=14)
    fig.tight_layout()
    fig.savefig(LAB_DIR / "regression_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_classification_plot(matrix, base_metrics, boosted_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=axes[0],
    )
    axes[0].set_title("Матрица ошибок\n(улучшенная модель)")
    axes[0].set_xlabel("Предсказанный класс")
    axes[0].set_ylabel("Истинный класс")

    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    x = range(len(metric_names))
    width = 0.35
    axes[1].bar(
        [i - width / 2 for i in x],
        [base_metrics[name] for name in metric_names],
        width=width,
        label="Логистическая регрессия",
        color="#7f8c8d",
    )
    axes[1].bar(
        [i + width / 2 for i in x],
        [boosted_metrics[name] for name in metric_names],
        width=width,
        label="Градиентный бустинг",
        color="#1f4e79",
    )
    axes[1].set_xticks(list(x), metric_names)
    axes[1].set_title("Сравнение метрик")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Классификация признака Transported", fontsize=14)
    fig.tight_layout()
    fig.savefig(LAB_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_dataset()

    print("Файл датасета:", DATASET_PATH)
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

    print("\nРазбиение для регрессии:")
    print("train:", len(x_reg_train), "test:", len(x_reg_test))

    linear_reg = LinearRegression()
    linear_reg.fit(x_reg_train, y_reg_train)
    linear_pred = linear_reg.predict(x_reg_test)

    boosted_reg = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
    )
    boosted_reg.fit(x_reg_train, y_reg_train)
    boosted_pred = boosted_reg.predict(x_reg_test)

    linear_reg_metrics = regression_metrics(y_reg_test, linear_pred)
    boosted_reg_metrics = regression_metrics(y_reg_test, boosted_pred)
    save_regression_plot(y_reg_test, linear_pred, boosted_pred, linear_reg_metrics, boosted_reg_metrics)

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

    print("\nРазбиение для классификации:")
    print("train:", len(x_cls_train), "test:", len(x_cls_test))

    base_clf = LogisticRegression(max_iter=3000)
    base_clf.fit(x_cls_train, y_cls_train)
    base_pred = base_clf.predict(x_cls_test)

    improved_clf = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
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
    save_classification_plot(matrix, base_cls_metrics, improved_cls_metrics)

    print("\nРегрессия: FoodCourt")
    print("Линейная регрессия:", linear_reg_metrics)
    print("Градиентный бустинг:", boosted_reg_metrics)

    print("\nКлассификация: Transported")
    print("Базовая логистическая регрессия:", base_cls_metrics)
    print("Градиентный бустинг:", improved_cls_metrics)
    print("Матрица ошибок:", matrix)

    print("\nСохранены файлы:")
    print(LAB_DIR / "regression_plot.png")
    print(LAB_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
