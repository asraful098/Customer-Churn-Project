
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def get_primary_model(preprocessor):
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42
            ))
        ]),
        "Decision Tree" : Pipeline([
            ('preprocessor',preprocessor),
            ('model',DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced'
            ))
        ])
        ,
        'SVC':Pipeline([
            ('preprocessor',preprocessor),
            ('model',SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])
        ,
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss'
            ))
        ])
    }

    return models


def quick_comprace(models, xtrain, ytrain, xtest, ytest):
    results = []

    for name, pipeline in models.items():
        pipeline.fit(xtrain, ytrain)

        pred = pipeline.predict(xtest)

        prob = pipeline.predict_proba(xtest)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(ytest, pred),
            "Precision": precision_score(
                ytest, pred, zero_division=0
            ),
            "Recall": recall_score(
                ytest, pred, zero_division=0
            ),
            "F1 Score": f1_score(
                ytest, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(ytest, prob)
        })

    return sorted(
        results,
        key=lambda x: x["F1 Score"],
        reverse=True
    )


if __name__ == "__main__":
    print("Primary models successfully configured")
