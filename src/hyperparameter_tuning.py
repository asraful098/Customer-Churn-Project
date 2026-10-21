from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


xgb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

grd_bc_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.8, 1.0],
    "model__min_samples_leaf": [1, 5, 10]
}

log_r_param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["lbfgs"],
    "model__class_weight": [None, "balanced"]
}

rnd_f_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", 0.8]
}

svc_param_grid = {
    "model__C": [0.1, 1, 10],
    "model__kernel": ["rbf", "linear"],
    "model__class_weight": [None, "balanced"]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

def tune_xgboostclassifier(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(steps=[
        ('processor',preprocessor),
        ('model',XGBClassifier(random_state=42,eval_metric="logloss"))
    ])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=xgb_param_grid,
        n_iter=15,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(xtrain,ytrain)
    return search

def tune_gradinent_bcla(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(steps=[
        ('processor',preprocessor),
        ('model',GradientBoostingClassifier(random_state=42))
    ])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=grd_bc_param_grid,
        n_iter=15,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(xtrain,ytrain)
    return search

def tune_logistic_reg(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(steps=[
        ('processor',preprocessor),
        ('model',LogisticRegression(random_state=42,max_iter=2000))
    ])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=log_r_param_grid,
        n_iter=10,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(xtrain,ytrain)
    return search

def tune_random_fcla(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(steps=[
        ('processor',preprocessor),
        ('model',RandomForestClassifier(random_state=42))
    ])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=rnd_f_param_grid,
        n_iter=15,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(xtrain,ytrain)
    return search

def tune_svc(preprocessor, xtrain, ytrain):
    pipeline = Pipeline(steps=[
        ('processor',preprocessor),
        ('model',SVC(random_state=42,probability=True))
    ])
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=svc_param_grid,
        n_iter=15,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42
    )
    search.fit(xtrain,ytrain)
    return search

if __name__ == "__main__":
    print("yes it's working")