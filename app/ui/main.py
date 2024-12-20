import streamlit as st
import json
import requests

st.write("""
# Kings League Dado Prediction
""")

st.sidebar.header('User Input Parameters')

# Función para capturar los datos de entrada del usuario
def user_input_features():
    team1 = st.sidebar.text_input("Team 1")
    team2 = st.sidebar.text_input("Team 2")
    dado_number = st.sidebar.number_input("Dice Number", min_value=1, max_value=6, step=1)
    minutes_after_dice_roll = st.sidebar.number_input("Minutes After Dice Roll", min_value=1, max_value=10, step=1)

    data = {
        'team1': team1,
        'team2': team2,
        'dado_number': dado_number,
        'minutes_after_dice_roll': minutes_after_dice_roll
    }

    return data

# Obtener los datos del usuario
df_dict = user_input_features()

# Botón para realizar la predicción
if st.button("Predict"):
    try:
        response = requests.post(
            url="http://127.0.0.1:8000/predict",
            data=json.dumps(df_dict),
            headers={"Content-Type": "application/json"}
        )
        st.write("Respuesta del servidor:", response.text)  # Ver qué se recibe
        response_data = response.json()
        st.write("Prediction Result:", response_data["prediction"])
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
    except json.JSONDecodeError:
        st.error("Invalid response from the prediction server.")
