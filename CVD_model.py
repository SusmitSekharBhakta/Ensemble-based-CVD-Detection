import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
# Load the data
data = pd.read_csv('heart.csv')
# Handling missing data
data = data.fillna(data.mean())
X = data.drop('HeartDiseaseorAttack', axis=1)
y = data['HeartDiseaseorAttack']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''model = StandardScaler()
new_data = model.fit_transform(X_test)
X_test = pd.DataFrame(new_data)
new_data = model.fit_transform(X_train)
X_train = pd.DataFrame(new_data)'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
sampler = SMOTE()
X_train, y_train = sampler.fit_resample(X_train, y_train)
X_test, y_test = sampler.fit_resample(X_test, y_test)
# Define the classifiers for the ensemble
classifiers = [
    ('decision_tree', DecisionTreeClassifier()),
    ('xgboost', XGBClassifier()),
    ('random_forest', RandomForestClassifier())
]
# model trainings
#ensemble = LogisticRegression(max_iter=1000, random_state= 64 )
#ensemble = DecisionTreeClassifier(random_state= 40)
#ensemble = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, random_state=100, max_features=5 )
#ensemble = RandomForestClassifier(n_estimators=200, random_state= 42, class_weight='balanced', max_depth=5)
#ensemble = AdaBoostClassifier(DecisionTreeClassifier(random_state= 40), n_estimators=200)
#ensemble = BaggingClassifier(base_estimator = DecisionTreeClassifier(random_state= 40), n_estimators = 200, random_state = 80)
#ensemble = StackingClassifier(estimators=classifiers)
#ensemble = GaussianNB(var_smoothing=1e-08)
#ensemble = XGBClassifier(random_state=42)
#ensemble = VotingClassifier(estimators=classifiers, voting='soft')
# Fit the models on the training data
ensemble.fit(X_train, y_train)
# Use the fitted model to make predictions on the test data
y_pred = ensemble.predict(X_test)
# Calculate metrics
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
