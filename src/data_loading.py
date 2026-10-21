import pandas as pd
from pathlib import Path

def load_data(path=None):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"data not found {path}\n",
            "put data path inside the data / folder"
        )
    
    df = pd.read_csv(path)
    return df
    
def data_stracture(df):
    print(f"Shape of Data\n {df.shape}")
    print(f"Data Info\n {df.info}")
    print(f"Clomnus:\n {df.columns.tolist()}")
    print(f"First 5 rows\n{df.head}")
    print(f"Data type\n {df.dtypes}")
    print(f"Missing values\n {df.isnull().sum()}")
    
if __name__ == "__main__":
    load_data('../data/orginal/WA_Fn-UseC_-Telco-Customer-Churn.csv')