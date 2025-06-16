import joblib

# Load model
model = joblib.load("model/random_forest.joblib")
status_encoder = joblib.load("model/status_encoder.joblib")

all_features = ["Application_mode", "Course", "Previous_qualification", "Previous_qualification_grade", "Mothers_occupation", "Admission_grade", "Debtor", "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "Age_at_enrollment", "Curricular_units_1st_sem_credited", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved", "Unemployment_rate", "GDP"]
numerical_features = ["Previous_qualification_grade", "Admission_grade", "Age_at_enrollment", "Curricular_units_1st_sem_credited", "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved", "Unemployment_rate", "GDP"]
categorical_features = list(set(all_features) - set(numerical_features))

# Fungsi untuk prediksi
def predict(input_df):
    input_df = input_df.copy()

    # Konversi fitur kategorik ke tipe kategori
    for col in categorical_features:
        input_df[col] = input_df[col].astype("category")

    # Prediksi
    y_pred = model.predict(input_df)

    # Decode hasil prediksi ke label
    decoded = status_encoder.inverse_transform(y_pred)

    return decoded[0]