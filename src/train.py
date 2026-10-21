from main import (
    best_model, best_model_name, best_threshold, best_f1_score, X
)

import joblib 
from pathlib import Path


model_path = Path('../models')
model_path.mkdir(exist_ok=True)


if best_model is not None:
    model_artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "validation_f1": best_f1_score,
        "feature_columns": X.columns.tolist()
    }

    joblib.dump(
        model_artifact,
        model_path / "best_model.pkl"
    )
    

if __name__ == '__main__':
    print(f"Best Model name is {best_model}")