import joblib
import pandas as pd

artifact = joblib.load('models/best_model.pkl')
model = artifact['model']
threshold = artifact['threshold']
feature_columns = artifact['feature_columns']

# পুরো dataset (যেখানে Churn column আছে, সেটি বাদ দিয়ে)
df = pd.read_csv('data/orginal/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

X = df[feature_columns]

probas = model.predict_proba(X)[:, 1]
df['churn_probability'] = probas
df['churn_prediction'] = (probas >= threshold).astype(int)

print(df['churn_prediction'].value_counts())
print(df[['customerID', 'churn_probability', 'churn_prediction']].head(20))

from sklearn.metrics import confusion_matrix, classification_report

y_true = df['Churn'].map({'Yes': 1, 'No': 0})
print(confusion_matrix(y_true, df['churn_prediction']))
print(classification_report(y_true, df['churn_prediction']))

print(artifact['model_name'])