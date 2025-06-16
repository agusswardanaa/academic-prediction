import streamlit as st
import pandas as pd
from src.mapping import application_mode_map, course_map, previous_qualification_map, mothers_occupation_map
from src.preprocessing import preprocess
from src.predict import predict

data = {}

st.set_page_config(
    page_title="Student's Dropout and Academic Success App",
    page_icon="🎓"
)
st.header("🎓 Student's Dropout and Academic Success App")

# Input baris pertama
col1, col2 = st.columns(2)

application_modes = {v: k for k, v in application_mode_map.items()}
with col1:
    selected = st.selectbox("Application Mode", list(application_modes.keys()))
    data["Application_mode"] = application_modes[selected]

courses = {v: k for k, v in course_map.items()}
with col2:
    selected = st.selectbox("Course", list(courses.keys()))
    data["Course"] = courses[selected]

# Input baris dua
col1, col2 = st.columns(2)

previous_qualifications = {v: k for k, v in previous_qualification_map.items()}
with col1:
    selected = st.selectbox("Previous Qualification", list(previous_qualifications.keys()))
    data["Previous_qualification"] = previous_qualifications[selected]

with col2:
    grade = st.number_input("Previous Qualification Grade (0-200)", min_value=0.0, max_value=200.0)
    data["Previous_qualification_grade"] = float(grade)


# Input baris tiga
col1, col2 = st.columns(2)

mothers_occupations = {v: k for k, v in mothers_occupation_map.items()}
with col1:
    selected = st.selectbox("Mothers Occupation", list(mothers_occupations.keys()))
    data["Mothers_occupation"] = mothers_occupations[selected]

with col2:
    grade = st.number_input("Admission Grade (0-200)", min_value=0.0, max_value=200.0)
    data["Admission_grade"] = float(grade)

# Input baris empat
col1, col2, col3 = st.columns(3)

with col1:
    selected = st.selectbox("Debtor", ["Yes", "No"])
    data["Debtor"] = 1 if selected == "Yes" else 0

with col2:
    selected = st.selectbox("Tuition Fees Up to Date", ["Yes", "No"])
    data["Tuition_fees_up_to_date"] = 1 if selected == "Yes" else 0

with col3:
    selected = st.selectbox("Gender", ["Male", "Female"])
    data["Gender"] = 1 if selected == "Male" else 0

# Input baris lima
col1, col2 = st.columns(2)

with col1:
    selected = st.selectbox("Scholarship Holder", ["Yes", "No"])
    data["Scholarship_holder"] = 1 if selected == "Yes" else 0

with col2:
    age = st.number_input("Age at Enrollment", step=1)
    data["Age_at_enrollment"] = float(age)

# Input baris enam
col1, col2, col3 = st.columns(3)

with col1:
    units = st.number_input("Curricular Units 1st Semester (Credited)", step=1)
    data["Curricular_units_1st_sem_credited"] = float(units)

with col2:
    units = st.number_input("Curricular Units 2nd Semester (Evaluations)", step=1)
    data["Curricular_units_2nd_sem_evaluations"] = float(units)

with col3:
    units = st.number_input("Curricular Units 2nd Semester (Approved)", step=1)
    data["Curricular_units_2nd_sem_approved"] = float(units)

# Input baris tujuh
col1, col2 = st.columns(2)

with col1:
    unemployment = st.number_input("Unemployment Rate")
    data["Unemployment_rate"] = float(unemployment)

with col2:
    gdp = st.number_input("GDP")
    data["GDP"] = float(gdp)

# View raw data
data = pd.DataFrame([data])
with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

# Predict button
if st.button('Predict'):
    new_data = preprocess(data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write(f"Predicted Student Status: {predict(new_data)}")