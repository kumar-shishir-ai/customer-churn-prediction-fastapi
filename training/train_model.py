import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTEENN
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from train_utils import DATA_FILE_PATH, MODEL_DIR, MODEL_PATH 

df = pd.read_csv(DATA_FILE_PATH)

X = df.drop(columns='Churn')
y = df.Churn.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Add in categorical column
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','tenure_group']
num_cols = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']

num_pipe = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

models = {

    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "model__C": [0.1, 1, 10]
        }
    },

    "decision_tree": {
        "model": DecisionTreeClassifier(),
        "params": {
            "model__max_depth": [3, 5, 10],
            "model__min_samples_split": [2, 5]
        }
    },

    "random_forest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10]
        }
    },

    "svm": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 1],
            "model__kernel": ['linear', 'rbf']
        }
    },

    "xgboost": {
        "model": XGBClassifier(eval_metric='logloss'),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.01, 0.1]
        }
    }
}

best_models = {}

for model_name, model_info in models.items():
    print(f"\n========== {model_name} ==========")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('model', model_info['model'])
    ])

    # GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=model_info['params'], cv=5, scoring='f1', n_jobs=-1)

    # Fit model
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Prediction
    y_pred = best_model.predict(X_test)

    # Store best model
    best_models[model_name] = best_model


os.makedirs(MODEL_DIR,exist_ok=True)
joblib.dump(best_models['xgboost'], MODEL_PATH)