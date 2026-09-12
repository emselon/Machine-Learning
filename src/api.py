from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float


def train_model() -> tuple[StandardScaler, SVC]:
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = SVC(kernel="rbf", C=1.0, gamma="scale")
    model.fit(X_train_scaled, y_train)
    return scaler, model


scaler, model = train_model()
app = FastAPI(title="Make Moons SVM API", version="1.0.0")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "SVM API is running"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    features = [[request.feature_1, request.feature_2]]
    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    decision_score = float(model.decision_function(scaled_features)[0])

    return {
        "prediction": prediction,
        "label": f"class_{prediction}",
        "decision_score": decision_score,
        "model": "RBF SVM",
    }