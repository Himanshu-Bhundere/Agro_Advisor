import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data():
    crop_data = pd.read_csv("dataset/Crop_recommendation.csv")
    fertilizer_data = pd.read_csv("dataset/Fertilizer_Prediction.csv")
    return crop_data, fertilizer_data

crop_data, fertilizer_data = load_data()

fertilizer_data.rename(columns={"Temparature": "Temperature"}, inplace=True)

label_encoders = {}
category_mappings = {}

for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    fertilizer_data[col] = le.fit_transform(fertilizer_data[col])
    label_encoders[col] = le
    category_mappings[col] = dict(enumerate(le.classes_))

X_crop = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_data['label']

crop_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(crop_pipeline, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
search.fit(X_crop, y_crop)
crop_model = search.best_estimator_

X_fert = fertilizer_data[['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y_fert = fertilizer_data['Fertilizer Name']

fert_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", RandomForestClassifier(random_state=42))
])

search_fert = RandomizedSearchCV(fert_pipeline, param_distributions=param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
search_fert.fit(X_fert, y_fert)
fert_model = search_fert.best_estimator_

with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(crop_model, f)

with open("models/fert_model.pkl", "wb") as f:
    pickle.dump(fert_model, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/category_mappings.pkl", "wb") as f:
    pickle.dump(category_mappings, f)