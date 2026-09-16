import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

st.set_page_config(
    layout="centered",
    page_icon="🩺",
    page_title="Breast Cancer Detection System"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "This tool predicts whether a tumor is **benign** or **malignant** "
        "based on 30 cell nuclei measurements, using a trained ML classification model."
    )
    st.caption("Model accuracy: ~96% on test data")  # replace with your real metric
    st.warning("For educational/demo purposes only — not a medical diagnosis.")
    st.divider()
    st.caption("Built by Saniya")

# ---------- Header ----------
st.title("🤖 AI-Based Breast Cancer Detection System")
st.caption("Built using an AI/ML classification model")

model = joblib.load("best_pipe.joblib")

st.divider()
st.subheader("🔬 Enter Patient Features")

# ---------- Form (batches inputs, avoids rerun on every keystroke) ----------
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Mean**")
        radius_mean = st.number_input("Radius Mean", 0.0, step=0.1, help="Mean distance from center to perimeter points")
        texture_mean = st.number_input("Texture Mean", 0.0, step=0.1)
        perimeter_mean = st.number_input("Perimeter Mean", 0.0, step=0.1)
        area_mean = st.number_input("Area Mean", 0.0, step=0.1)
        smoothness_mean = st.number_input("Smoothness Mean", 0.0, step=0.001, format="%.4f")
        compactness_mean = st.number_input("Compactness Mean", 0.0, step=0.001, format="%.4f")
        concavity_mean = st.number_input("Concavity Mean", 0.0, step=0.001, format="%.4f")
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, step=0.001, format="%.4f")
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, step=0.001, format="%.4f")
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, step=0.001, format="%.4f")

    with col2:
        st.markdown("**SE**")
        radius_se = st.number_input("Radius SE", 0.0, step=0.01)
        texture_se = st.number_input("Texture SE", 0.0, step=0.01)
        perimeter_se = st.number_input("Perimeter SE", 0.0, step=0.01)
        area_se = st.number_input("Area SE", 0.0, step=0.1)
        smoothness_se = st.number_input("Smoothness SE", 0.0, step=0.0001, format="%.5f")
        compactness_se = st.number_input("Compactness SE", 0.0, step=0.0001, format="%.5f")
        concavity_se = st.number_input("Concavity SE", 0.0, step=0.0001, format="%.5f")
        concave_points_se = st.number_input("Concave Points SE", 0.0, step=0.0001, format="%.5f")
        symmetry_se = st.number_input("Symmetry SE", 0.0, step=0.0001, format="%.5f")
        fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, step=0.0001, format="%.5f")

    with col3:
        st.markdown("**Worst**")
        radius_worst = st.number_input("Radius Worst", 0.0, step=0.1)
        texture_worst = st.number_input("Texture Worst", 0.0, step=0.1)
        perimeter_worst = st.number_input("Perimeter Worst", 0.0, step=0.1)
        area_worst = st.number_input("Area Worst", 0.0, step=0.1)
        smoothness_worst = st.number_input("Smoothness Worst", 0.0, step=0.001, format="%.4f")
        compactness_worst = st.number_input("Compactness Worst", 0.0, step=0.001, format="%.4f")
        concavity_worst = st.number_input("Concavity Worst", 0.0, step=0.001, format="%.4f")
        concave_points_worst = st.number_input("Concave Points Worst", 0.0, step=0.001, format="%.4f")
        symmetry_worst = st.number_input("Symmetry Worst", 0.0, step=0.001, format="%.4f")
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, step=0.001, format="%.4f")

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])

    with st.spinner("Analyzing features..."):
        prediction = model.predict(input_data)[0]
        # Optional confidence, if the pipeline supports it
        try:
            proba = model.predict_proba(input_data)[0]
            confidence = round(max(proba) * 100, 1)
        except AttributeError:
            confidence = None

    st.divider()
    st.subheader("📋 Result")

    if prediction == 0:
        st.error("⚠️ Malignant Tumor Detected")
    else:
        st.success("✅ Benign Tumor Detected")

    if confidence is not None:
        st.metric("Model Confidence", f"{confidence}%")

    # Download the input + result as a record
    result_df = pd.DataFrame(input_data, columns=[
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ])
    result_df["prediction"] = "Malignant" if prediction == 0 else "Benign"
    st.download_button(
        "⬇️ Download this result as CSV",
        result_df.to_csv(index=False),
        file_name="prediction_result.csv",
        mime="text/csv"
    )