import streamlit as st
import pandas as pd
import joblib

rf = joblib.load("/app/models/random_forest_model.joblib")

st.title("🛍️ Customer Segmentation")

st.write(
    "Entrez les informations RFM du client "
    "pour prédire son segment."
)


# Saisie des données RFM
recency = st.number_input(
    "Recency",
    min_value=0.0,
    value=10.0
)

frequency = st.number_input(
    "Frequency",
    min_value=0.0,
    value=5.0
)

monetary = st.number_input(
    "Monetary",
    min_value=0.0,
    value=1000.0
)


# Dictionnaire des segments
segment_names = {
    0: "VIP",
    1: "Inactive Clients",
    2: "Regular Clients"
}


# Bouton de prédiction
if st.button("Prédire le segment"):

    new_client = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    prediction = rf.predict(new_client)[0]

    segment = segment_names[prediction]

    st.subheader("Résultat")

    st.write("**Cluster prédit :**", prediction)
    st.write("**Segment :**", segment)

    st.subheader("Informations RFM")

    st.dataframe(new_client)