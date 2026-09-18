import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# --------------------------------------------------
# Load trained model and preprocessing objects
# --------------------------------------------------

model = joblib.load("osteoporosis_xgb_model.pkl")
scaler = joblib.load("osteoporosis_scaler.pkl")
model_features = joblib.load("osteoporosis_features.pkl")

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Osteoporosis Risk Assessment",
    page_icon="🦴",
    layout="centered"
)

# --------------------------------------------------
# Title and introduction
# --------------------------------------------------

st.title("🦴 Osteoporosis Risk Assessment")

st.write(
    "A prototype AI-based clinical decision-support tool "
    "for estimating osteoporosis risk from patient characteristics."
)

st.info(
    "This is an AI/ML prototype for educational and demonstration "
    "purposes. It is not a diagnostic tool and should not replace "
    "clinical assessment or professional medical advice."
)

# --------------------------------------------------
# Patient Information
# --------------------------------------------------

st.header("1. Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=None,
    placeholder="Enter age"
)

gender = st.selectbox(
    "Gender",
    ["", "Female", "Male"]
)

if gender == "Female":

    hormonal_changes = st.selectbox(
        "Hormonal Changes",
        ["", "Premenopausal", "Postmenopausal"]
    )

else:

    hormonal_changes = "Not Applicable"

family_history = st.selectbox(
    "Family History of Osteoporosis",
    ["", "No", "Yes"]
)

body_weight = st.selectbox(
    "Body Weight",
    ["", "Normal", "Underweight", "Overweight"]
)

calcium = st.selectbox(
    "Calcium Intake",
    ["", "Adequate", "Low"]
)

vitamin_d = st.selectbox(
    "Vitamin D Intake",
    ["", "Insufficient", "Sufficient"]
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["", "Active", "Sedentary"]
)

smoking = st.selectbox(
    "Smoking",
    ["", "No", "Yes"]
)

alcohol = st.selectbox(
    "Alcohol Consumption",
    ["", "No", "Yes", "Not Reported"]
)

medical_conditions = st.selectbox(
    "Medical Conditions",
    ["", "None", "Rheumatoid Arthritis", "Not Reported"]
)


prior_fractures = st.selectbox(
    "Prior Fractures",
    ["", "No", "Yes"]
)

# --------------------------------------------------
# Risk Assessment
# --------------------------------------------------

if st.button("Assess Osteoporosis Risk"):

    required_fields = [
        gender,
        hormonal_changes,
        family_history,
        body_weight,
        calcium,
        vitamin_d,
        physical_activity,
        smoking,
        alcohol,
        medical_conditions,
        prior_fractures
    ]

    if age is None or any(field == "" for field in required_fields):

        st.warning(
            "Please complete all patient fields before assessing risk."
        )

    else:

        # ----------------------------------------------
        # Create patient dataframe
        # ----------------------------------------------

        patient = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Hormonal Changes": hormonal_changes,
            "Family History": family_history,
            "Body Weight": body_weight,
            "Calcium Intake": calcium,
            "Vitamin D Intake": vitamin_d,
            "Physical Activity": physical_activity,
            "Smoking": smoking,
            "Alcohol Consumption": alcohol,
            "Medical Conditions": medical_conditions,
            "Medications": "Not Reported",
            "Prior Fractures": prior_fractures
        }])

        # ----------------------------------------------
        # Encode categorical variables
        # ----------------------------------------------

        patient_encoded = pd.get_dummies(
            patient,
            drop_first=False
        )

        # ----------------------------------------------
        # Align exactly with training features
        # ----------------------------------------------

        patient_processed = pd.DataFrame(
            0.0,
            index=[0],
            columns=model_features,
            dtype=float
        )

        for feature in model_features:

            if feature in patient_encoded.columns:

                patient_processed.loc[0, feature] = float(
                    patient_encoded.loc[0, feature]
                )

        # ----------------------------------------------
        # Scale Age
        # ----------------------------------------------

        patient_processed["Age"] = scaler.transform(
            patient_processed[["Age"]]
        ).ravel()

        # ----------------------------------------------
        # Model prediction
        # ----------------------------------------------

        probability = model.predict_proba(
            patient_processed
        )[0, 1]

        prediction = model.predict(
            patient_processed
        )[0]

        # ----------------------------------------------
        # Risk category
        # ----------------------------------------------

        if probability < 0.30:

            risk_category = "Low Risk"

        elif probability < 0.60:

            risk_category = "Moderate Risk"

        else:

            risk_category = "High Risk"

        # ----------------------------------------------
        # Display prediction
        # ----------------------------------------------

        st.header("2. AI Risk Assessment")

        st.metric(
            "Estimated Osteoporosis Probability",
            f"{probability * 100:.1f}%"
        )

        if risk_category == "Low Risk":

            st.success(risk_category)

        elif risk_category == "Moderate Risk":

            st.warning(risk_category)

        else:

            st.error(risk_category)

        st.write(
            "**Model classification:** ",
            "Osteoporosis" if prediction == 1
            else "No Osteoporosis"
        )

        st.caption(
            "Risk categories are prototype thresholds for "
            "demonstration and are not validated clinical "
            "diagnostic thresholds."
        )

        # --------------------------------------------------
        # SHAP Explanation
        # --------------------------------------------------

        st.header("3. Why did the model make this prediction?")

        st.write(
            "SHAP (SHapley Additive exPlanations) is used to "
            "show which patient characteristics contributed "
            "most to the model's prediction."
        )

        try:

            # ----------------------------------------------
            # Calculate SHAP values
            # ----------------------------------------------

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                patient_processed
            )

            if isinstance(shap_values, list):

                shap_for_patient = shap_values[1][0]

            else:

                shap_for_patient = shap_values[0]

            # ----------------------------------------------
            # Create explanation dataframe
            # ----------------------------------------------

            explanation_df = pd.DataFrame({
                "Feature": model_features,
                "SHAP Value": shap_for_patient,
                "Patient Value": patient_processed.iloc[0].values
            })

            explanation_df["Absolute SHAP"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute SHAP",
                ascending=False
            )

            # ----------------------------------------------
            # Top contributing factors
            # ----------------------------------------------

            st.subheader("Top contributing factors")

            top_features = explanation_df.head(5)

            for _, row in top_features.iterrows():

                feature = row["Feature"]
                shap_value = row["SHAP Value"]
                patient_value = row["Patient Value"]

                if "_" in feature:

                    base_feature, category = feature.split(
                        "_", 1
                    )

                    if patient_value == 1:

                        display_feature = (
                            f"{base_feature}: {category}"
                        )

                    else:

                        display_feature = (
                            f"{base_feature}: "
                            f"{category} not present"
                        )

                else:

                    display_feature = feature

                if shap_value > 0:

                    st.markdown(
                        f'<span style="color:red; '
                        f'font-size:20px;">↑</span> '
                        f'**{display_feature}** — '
                        f'contributed toward higher predicted risk',
                        unsafe_allow_html=True
                    )

                elif shap_value < 0:

                    st.markdown(
                        f'<span style="color:green; '
                        f'font-size:20px;">↓</span> '
                        f'**{display_feature}** — '
                        f'contributed toward lower predicted risk',
                        unsafe_allow_html=True
                    )

                else:

                    st.write(
                        f"• **{display_feature}** — "
                        f"minimal contribution"
                    )

            # ----------------------------------------------
            # SHAP bar chart
            # ----------------------------------------------

            st.subheader("Model feature contributions")

            chart_df = explanation_df.head(10)[
                ["Feature", "SHAP Value"]
            ].set_index("Feature")

            st.bar_chart(chart_df)

        except Exception:

            st.warning(
                "SHAP explanation could not be generated "
                "for this prediction."
            )

        # --------------------------------------------------
        # DXA consideration
        # --------------------------------------------------

        st.header("4. DXA Assessment Consideration")

        st.write(
            "DXA (Dual-energy X-ray Absorptiometry) is commonly "
            "used to measure bone mineral density and assess "
            "bone health."
        )

        st.write(
            "The AI prediction above does not independently "
            "determine whether a DXA scan is clinically required."
        )

        # ----------------------------------------------
        # Prototype DXA guidance
        # ----------------------------------------------

        dxa_factors = []

        if age >= 65:

            dxa_factors.append(
                "Age ≥65 years"
            )

        if prior_fractures == "Yes":

            dxa_factors.append(
                "History of prior fracture"
            )

        if family_history == "Yes":

            dxa_factors.append(
                "Family history of osteoporosis"
            )

        if hormonal_changes == "Postmenopausal":

            dxa_factors.append(
                "Postmenopausal status"
            )

        if body_weight == "Underweight":

            dxa_factors.append(
                "Underweight status"
            )

        if medical_conditions == "Rheumatoid Arthritis":

            dxa_factors.append(
                "Rheumatoid arthritis"
            )

        if dxa_factors:

            st.warning(
                "Clinical assessment for bone health and possible "
                "DXA testing may be appropriate based on the "
                "patient's risk factors."
            )

            st.write(
                "Relevant factors identified:"
            )

            for factor in dxa_factors:

                st.write(
                    f"• {factor}"
                )

        else:

            st.info(
                "No major prototype DXA consideration factors "
                "were identified from the information entered. "
                "Clinical assessment should still determine "
                "whether further evaluation is appropriate."
            )

        st.caption(
            "This DXA section is a prototype decision-support "
            "layer and is not a substitute for clinical guidelines "
            "or professional assessment."
        )

        # --------------------------------------------------
        # Final disclaimer
        # --------------------------------------------------

        st.divider()

        st.caption(
            "Prototype only | Not for diagnosis or treatment. "
            "AI predictions should be interpreted alongside "
            "clinical history, examination and appropriate "
            "clinical investigations."
        )
