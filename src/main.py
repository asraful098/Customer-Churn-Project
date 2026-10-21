from pathlib import Path
import importlib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)
from sklearn.base import clone
from sklearn.model_selection import train_test_split


loder = importlib.import_module('data_loading')
data_process = importlib.import_module('data_preprocessing_pipeline')
primary_model = importlib.import_module('primary_model')
from hyperparameter_tuning import (
    tune_logistic_reg,
    tune_gradinent_bcla,
    tune_xgboostclassifier,
    tune_random_fcla,
    tune_svc
)

output_dir = Path('../outputs')
output_dir.mkdir(exist_ok=True)

df = loder.load_data('../data/orginal/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# data_s = loder.data_stracture(df)
# print(data_s)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

target = 'Churn'
X = df.drop(columns=["customerID",target])
y = df[target].map({
    "Yes": 1,
    "No": 0
})


X_train, X_test, y_train, y_test = data_process.split_data(X,y)


processor = data_process.process_data(X_train)

models = primary_model.get_primary_model(processor)
basline_models_result = primary_model.quick_comprace(
    models,
    X_train,
    y_train,
    X_test,
    y_test  
)

baseline_df = pd.DataFrame(basline_models_result)
baseline_df.to_csv("../outputs/baseline_model_comparison.csv",index=False)


tuning_functions = {
    "XGBoost": tune_xgboostclassifier,
    "Gradient Boosting": tune_gradinent_bcla,
    "Logistic Regression": tune_logistic_reg,
    "Random Forest": tune_random_fcla,
    "SVC": tune_svc
}

results = []
best_model = None
best_model_name = None
best_threshold = None
best_f1_score = float("-inf")


report_path = Path('../report')
report_path.mkdir(exist_ok=True)

X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

for model_name, tune_function in tuning_functions.items():
    print(f'Training {model_name}')
    
    search = tune_function(
        clone(processor),
        X_subtrain,
        y_subtrain
    )
    
    fited_model = search.best_estimator_
    
    if hasattr(fited_model, 'predict_proba'):
        value_score = fited_model.predict_proba(X_val)[:, 1]
    else:
        value_score = fited_model.decision_function(X_val)
        
    if hasattr(fited_model, 'predict_proba'):
        thresholds = np.arange(0.20, 0.81, 0.05)
    else:
        thresholds = np.unique(
            np.quantile(value_score, np.linspace(0.05, 0.95, 19))
        )
    
    threshold_results = []
    
    for thershlod in thresholds:
        val_pred = (value_score >= thershlod).astype(int)
        
        threshold_results.append({
            'Threshold' : float(thershlod),
            'precision' : precision_score(
                y_val,val_pred, zero_division=0  
            ),
            "Recall": recall_score(
                y_val, val_pred, zero_division=0
            ),
            "F1_Score": f1_score(
                y_val, val_pred, zero_division=0
            )
        })
        
    threshold_df = pd.DataFrame(threshold_results)
        
    min_precision = 0.50
    valid = threshold_df[threshold_df['precision'] >= min_precision]

    if not valid.empty:
        best_threshold_row = valid.loc[valid['Recall'].idxmax()]
    else:
        best_threshold_row = threshold_df.loc[threshold_df['F1_Score'].idxmax()]
        
    best_threshold = best_threshold_row['Threshold']
    
    if hasattr(fited_model, "predict_proba"):
        test_scores = fited_model.predict_proba(X_test)[:, 1]
    else:
        test_scores = fited_model.decision_function(X_test)

    y_pred = (test_scores >= best_threshold).astype(int)
        
    metrics = {
        "Model": model_name,
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, test_scores),
        "Best_CV_ROC_AUC": search.best_score_,
        "Best_Params": str(search.best_params_)
    }

    results.append(metrics)
    
    validation_f1 = best_threshold_row["F1_Score"]
    
    if validation_f1 > best_f1_score:
        best_f1_score = validation_f1
        best_model = fited_model
        best_model_name = model_name
        best_threshold = float(best_threshold)
        
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Best_CV_ROC_AUC",
    ascending=False
)

results_df.to_csv(report_path/'model_comparison.csv')


