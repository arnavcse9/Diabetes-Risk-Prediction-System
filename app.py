"""Diabetes Risk Predictor: Streamlit web version of the PyQt5 desktop app.

Educational demo only. Not medical advice.
Model: Support Vector Machine (linear kernel) trained on the Pima Indians
Diabetes dataset (UCI / NIDDK). The model is retrained from pima.csv when the
app starts, so it never depends on a pickle file built with another library
version.
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

DATA_FILE = Path(__file__).parent / "pima.csv"

# Same five features and column letters used in diabetes.py
FEATURES = ["B", "C", "D", "E", "F"]
TARGET = "I"


def build_model():
    """Train the SVC exactly as diabetes.py does and report held-out accuracy."""
    data = pd.read_csv(DATA_FILE)
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=101
    )
    model = SVC(kernel="linear").fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    baseline = float((y_test == 0).mean())  # always guessing "no diabetes"
    return model, accuracy, baseline, len(y_test)


def predict(model, glucose, blood_pressure, skin_thickness, insulin, bmi):
    row = pd.DataFrame(
        [[glucose, blood_pressure, skin_thickness, insulin, bmi]],
        columns=FEATURES,
    )
    return int(model.predict(row)[0])


@st.cache_resource
def load_model():
    return build_model()


def main():
    st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
    st.title("Diabetes Risk Predictor")
    st.caption(
        "Training project by Arnav Sharma (JSRRB Technologies). "
        "Educational demo only. Not medical advice."
    )

    model, accuracy, baseline, n_test = load_model()

    st.subheader("Enter measurements")
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input(
            "Plasma glucose (mg/dL)", 0.0, 300.0, 120.0, step=1.0,
            help="2 hours into an oral glucose tolerance test.",
        )
        blood_pressure = st.number_input(
            "Diastolic blood pressure (mm Hg)", 0.0, 200.0, 70.0, step=1.0
        )
        skin_thickness = st.number_input(
            "Triceps skin fold thickness (mm)", 0.0, 120.0, 20.0, step=1.0
        )
    with col2:
        insulin = st.number_input(
            "Serum insulin (mu U/ml)", 0.0, 900.0, 80.0, step=1.0,
            help="2-hour serum insulin.",
        )
        bmi = st.number_input(
            "Body mass index (kg/m²)", 0.0, 80.0, 25.0, step=0.1,
            help="Weight in kg divided by height in metres squared.",
        )

    if st.button("Check risk", type="primary"):
        result = predict(model, glucose, blood_pressure, skin_thickness, insulin, bmi)
        if result == 1:
            st.warning(
                "The model estimates a **higher risk** pattern. "
                "This is not a diagnosis. Please talk to a doctor."
            )
        else:
            st.success(
                "The model estimates a **lower risk** pattern. "
                "This is not a diagnosis and does not replace a medical check-up."
            )

    with st.expander("About this model"):
        st.markdown(
            f"""
- **Model:** Support Vector Machine (linear kernel) built with scikit-learn.
- **Data:** Pima Indians Diabetes dataset (UCI / NIDDK), 768 records.
- **Held-out accuracy:** {accuracy:.1%} on a {n_test}-record test split
  (always guessing "no diabetes" would score {baseline:.1%}).
- **Limits:** small, old dataset; five input features only; not validated for
  medical use.
- **Privacy:** no names or inputs are stored.
"""
        )


if __name__ == "__main__":
    main()
