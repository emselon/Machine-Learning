"""Huấn luyện và trực quan hóa hai SVM trên dữ liệu make_moons."""

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_svm_decision_boundary(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    title: str,
) -> None:
    """Vẽ dữ liệu, decision boundary, margin và support vectors."""
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    decision_values = model.decision_function(grid).reshape(grid_x.shape)

    ax.contour(
        grid_x,
        grid_y,
        decision_values,
        levels=[-1, 0, 1],
        colors=["#f59e0b", "#111827", "#f59e0b"],
        linestyles=["--", "-", "--"],
        linewidths=[1.2, 1.8, 1.2],
    )
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.85,
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=130,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.8,
        label="Support vectors",
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)


def main() -> None:
    """Tạo dữ liệu, train hai model, in báo cáo và lưu hình kết quả."""
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models: Dict[str, SVC] = {
        "Linear SVM": SVC(kernel="linear", C=1.0),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }

    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    results: Dict[str, Tuple[float, int]] = {}

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        support_vector_count = len(model.support_vectors_)
        results[model_name] = (accuracy, support_vector_count)

        print(f"\n{'=' * 60}\n{model_name}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Support vectors: {support_vector_count}/{len(X_train_scaled)}")
        print("Classification report:")
        print(classification_report(y_test, predictions))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

        plot_svm_decision_boundary(
            model,
            X_train_scaled,
            y_train,
            axes[0 if model_name == "Linear SVM" else 1],
            (
                f"{model_name}\n"
                f"Accuracy: {accuracy:.2f} | SV: {support_vector_count}"
            ),
        )

    fig.suptitle("Linear SVM vs RBF SVM trên make_moons", fontsize=14)
    fig.tight_layout()
    output_path = "outputs/svm_linear_vs_rbf.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nĐã lưu hình kết quả tại: {output_path}")
    print("\nKết luận:")
    for model_name, (accuracy, support_vector_count) in results.items():
        if model_name == "Linear SVM":
            conclusion = "khó tách dữ liệu moons phi tuyến"
        else:
            conclusion = "kernel trick tách dữ liệu phi tuyến tốt hơn"
        print(
            f"{model_name}: accuracy={accuracy:.2f}, "
            f"SV={support_vector_count}/{len(X_train_scaled)} -> {conclusion}"
        )


if __name__ == "__main__":
    main()