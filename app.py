import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('🦶Foot Measurement Prediction App')
st.sidebar.title("About the Model and Leprosy")

st.sidebar.write("""
This application uses a linear regression model to predict foot measurements based on patient characteristics.

**Leprosy:**
Leprosy is a chronic infectious disease caused by the bacterium *Mycobacterium leprae*. It primarily affects the skin, peripheral nerves, upper respiratory tract, eyes, and testes. Leprosy is curable with multidrug therapy (MDT).

**The Model:**
The model was trained on a dataset containing foot measurements and patient information, including age, gender, grade of leprosy, and diabetes status. It aims to predict various foot measurements based on these input features.
""")


# Input widgets
age = st.number_input('Patient Age', min_value=0, max_value=120, value=50)
gender = st.selectbox('Patient Gender', ['Male', 'Female'])
grade = st.selectbox('Grade', ['Grade I', 'Grade II'])
diabetes = st.selectbox('Diabeties', ['No', 'Yes'])

if st.button('Predict Foot Measurements'):
    # Load the trained model
    model = joblib.load('linear_regression_model.joblib')

    # Prepare data for prediction
    sample_data = {
        "Patient Age": [age],
        "Patient Gender_Female": [1 if gender == 'Female' else 0],
        "Patient Gender_Male": [1 if gender == 'Male' else 0],
        "Grade_Grade I": [1 if grade == 'Grade I' else 0],
        "Grade_Grade II": [1 if grade == 'Grade II' else 0],
        "Diabeties_No": [1 if diabetes == 'No' else 0],
        "Diabeties_Yes": [1 if diabetes == 'Yes' else 0]
    }
    sample_df = pd.DataFrame(sample_data)

    # Predict
    predicted_measurements = model.predict(sample_df)

    # Display predictions
    st.subheader('Predicted Foot Measurements:')
    st.write(f"Length of Foot Left: {predicted_measurements[0][0]:.2f}")
    st.write(f"Length of Foot Right: {predicted_measurements[0][1]:.2f}")
    st.write(f"Ball Girth Left: {predicted_measurements[0][2]:.2f}")
    st.write(f"Ball Girth Right: {predicted_measurements[0][3]:.2f}")
    st.write(f"In Step Left: {predicted_measurements[0][4]:.2f}")
    st.write(f"In Step Right: {predicted_measurements[0][5]:.2f}")

    # Add some basic visualizations
    st.subheader('Visualizations of Predicted Measurements:')

    measurements = ['Length of Foot Left', 'Length of Foot Right', 'Ball Girth Left', 'Ball Girth Right', 'In Step Left', 'In Step Right']
    predicted_values = predicted_measurements[0]

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=measurements, y=predicted_values, ax=ax)
    ax.set_title('Predicted Foot Measurements')
    ax.set_ylabel('Measurement (cm)')
    ax.set_xticklabels(measurements, rotation=45, ha='right')
    st.pyplot(fig)