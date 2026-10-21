from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def process_data(X):
    
    numeric_col = X.select_dtypes(include=['number']).columns.tolist()
    boolen_col = X.select_dtypes(include=['bool']).columns.tolist()
    categorical_col = X.select_dtypes(include=['object','category']).columns.tolist()
    
    numeric_pipe = Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps=[
            ('imputure',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    
    boolen_pipeline = Pipeline(
        steps=[
            ('imputure', SimpleImputer(strategy='most_frequent'))
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num',numeric_pipe,numeric_col),
            ('cat',categorical_pipeline, categorical_col),
            ('bool',boolen_pipeline, boolen_col)
        ]
    )
    
    return preprocessor


if __name__ =="__main__":
    print('data preprocessing working')
