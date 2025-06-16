import joblib

# Load model
scaler = joblib.load("model/scaler.joblib")

numerical_features = ["Previous_qualification_grade", "Admission_grade", "Age_at_enrollment", "Curricular_units_1st_sem_credited", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved", "Unemployment_rate", "GDP"]

# Standardisasi input data
def preprocess(data):
    # Sesuaikan urutan kolom dengan model training
    feature_names = joblib.load("model/feature_names.joblib")
    data = data[feature_names]
    
    numerical_data = data[numerical_features]
    scaled = scaler.transform(numerical_data)
    data[numerical_features] = scaled

    return data