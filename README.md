# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service. The project covers data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, threshold optimization, and an interactive Streamlit web application.

**Live Demo:** [Customer-Churn-prediction-app](https://customer-churn-project-as.streamlit.app/)
**GitHub Repository:** [Customer-Churn-Project](https://github.com/asraful098/Customer-Churn-Project)

## Project Overview

Customer churn prediction helps businesses identify customers who may discontinue their services. By analyzing customer demographics, subscribed services, contracts, billing information, and other features, machine learning models can estimate churn risk and support customer retention strategies.

This project uses the IBM Telco Customer Churn dataset to build and evaluate multiple classification models.

## Objectives

- Explore and understand customer churn patterns.
- Clean and preprocess the dataset.
- Perform exploratory data analysis and feature preparation.
- Train and compare multiple machine learning classification models.
- Tune hyperparameters using cross-validation.
- Optimize classification thresholds to balance precision and recall.
- Evaluate models using appropriate classification metrics.
- Deploy an interactive prediction application using Streamlit.

## Dataset

**Dataset:** IBM Telco Customer Churn — `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information, subscribed services, account details, and a target column indicating whether a customer churned.

The target variable is `Churn`:

- `Yes` — The customer churned.
- `No` — The customer stayed.

The `customerID` column is an identifier rather than a predictive feature, so it is excluded from model training.

**Important:** The dataset is not necessarily included in this GitHub repository. Download it from an authorized source and place it in the expected local data directory before running the training workflow.

## Machine Learning Workflow

1. **Data loading:** Load the customer churn dataset using Pandas.
2. **Data cleaning:** Handle missing values, correct data types, and prepare the target variable.
3. **Exploratory data analysis:** Examine churn distribution, customer characteristics, and relationships between features and churn.
4. **Preprocessing:** Apply numerical imputation and scaling, categorical imputation and one-hot encoding, and suitable transformations.
5. **Data splitting:** Separate training and test data to evaluate model generalization.
6. **Model training:** Train several classification algorithms.
7. **Hyperparameter tuning:** Search for promising configurations using cross-validation.
8. **Threshold optimization:** Evaluate different decision thresholds using validation predictions.
9. **Model evaluation:** Compare classification metrics and ROC-AUC.
10. **Deployment:** Provide an interactive interface for customer churn predictions using Streamlit.

## Models Evaluated

The project compares the following classification algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Support Vector Classifier (SVC)

Model selection should depend on the business objective. For example, a company that wants to identify as many potential churners as possible may prioritize recall, while a company seeking a balance between precision and recall may prioritize F1-score.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Percentage of correct predictions |
| Precision | How many predicted churners actually churn |
| Recall | How many actual churners the model identifies |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish churners from non-churners |

### Example Results

The following results are from a recent model comparison in this project. They are evaluation results from that run, not a guarantee of future performance.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.7970 | 0.5916 | 0.7534 | 0.6627 | 0.8615 |
| Gradient Boosting | 0.7984 | 0.5953 | 0.7453 | 0.6619 | 0.8596 |
| Logistic Regression | 0.7984 | 0.6042 | 0.6917 | 0.6450 | 0.8620 |
| Random Forest | 0.7928 | 0.5832 | 0.7614 | 0.6605 | 0.8563 |
| SVC | 0.8070 | 0.6241 | 0.6810 | 0.6513 | 0.8524 |

In this comparison, XGBoost achieved the highest F1-score, while SVC achieved the highest accuracy and Logistic Regression achieved the highest ROC-AUC. These metrics use different decision thresholds where optimized, so the best model depends on the intended business outcome.

## Threshold Optimization

Classification models commonly use a default threshold of `0.50` when converting probabilities into class predictions. However, adjusting this threshold can change the balance between precision and recall.

This project evaluates candidate thresholds on validation predictions to find a suitable operating point. The selected threshold should be saved alongside the fitted model and evaluated on an untouched test set before being used in production.

**Note:** Threshold tuning changes class predictions and can affect accuracy, precision, recall, and F1-score. It does not change ROC-AUC when calculated from the same underlying prediction scores.

## Streamlit Web Application

Try the deployed application:

**[Open Customer Churn Prediction App](https://customer-churn-project-as.streamlit.app/)**

The application provides a user-facing interface for entering customer information and generating churn predictions. Availability depends on the deployment being active and its configured dependencies and model files.

## Tech Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn, XGBoost
- **Model evaluation:** Cross-validation, classification metrics, ROC-AUC, threshold optimization
- **Web application:** Streamlit
- **Experiment tracking:** MLflow, where configured
- **Development tools:** Jupyter Notebook, VS Code, Git, GitHub

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/asraful098/Customer-Churn-Project.git
cd Customer-Churn-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

If the repository contains `requirements-project.txt`, run:

```bash
pip install -r requirements-project.txt
```

If it does not, install the project's required packages according to the imports and dependency versions used by the code.

### 4. Add the dataset

Download the IBM Telco Customer Churn CSV and place it in the input directory expected by the project's data-loading code. Check the configuration or data-loading script for the exact path.

### 5. Run the project

Run the training or pipeline script from the project root, using the script name and commands provided in the repository. For example, if `src/main.py` is the entry point:

```bash
python src/main.py
```

### 6. Launch Streamlit

If `app.py` is located in the project root:

```bash
streamlit run app.py
```

If the application file is stored elsewhere, replace `app.py` with its actual path.

## Suggested Repository Structure

The exact directory structure may differ from this example.

```text
Customer-Churn-Project/
├── data/
│   ├── raw/
│   └── process/
├── notebooks/
├── src/
├── models/
├── app.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Keep large generated datasets, virtual environments, credentials, experiment artifacts, and local-only model files out of Git unless they are intentionally managed through an appropriate storage solution.

## Future Improvements

- Improve model performance through further feature engineering and tuning.
- Add model explainability using SHAP or similar tools.
- Monitor model performance and data drift.
- Track reproducible experiments and model versions.
- Add automated testing and a reproducible training pipeline.
- Evaluate calibration and business-specific churn intervention costs.
- Improve the deployed application's validation, reliability, and user experience.

## Author

**Sheikh Md. Asraful Islam Robin**  
CSE Student | Machine Learning Enthusiast

- GitHub: [Asraful098](https://github.com/asraful098)
- Portfolio: [Asraful | AI/ML Developer](https://asraful098.github.io/asaful-site/)
- Project Repository: [Customer-Churn-Project](https://github.com/asraful098/Customer-Churn-Project)
- Live Application: [Customer-Churn-prediction-app](https://customer-churn-project-as.streamlit.app/)

## License

This project uses the license included in the repository. The project code's license does not automatically apply to the dataset or third-party libraries; check their respective terms separately.
