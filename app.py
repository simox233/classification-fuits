import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Configuration de la page
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Fonction pour charger et préparer l'image
def read_image(uploaded_file):
    # Créer un fichier temporaire pour stocker l'image
    temp_file = f"temp_image.{uploaded_file.name.split('.')[-1]}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Charger et prétraiter l'image
    img = Image.open(temp_file).convert("RGB")  # Convertir en RGB pour s'assurer qu'il n'y a que 3 canaux
    img = img.resize((32, 32))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # Supprimer le fichier temporaire
    os.remove(temp_file)
    return x


def main():
    st.sidebar.title("Fruit Classifier")
    st.sidebar.write("Upload an image of a fruit to classify it!")

    # Charger le modèle
    @st.cache_resource
    def load_classifier():
        return load_model('model.h5')
    
    try:
        model = load_classifier()
    except Exception as e:
        st.error("Failed to load model. Please make sure 'model.h5' exists in the current directory.")
        st.stop()

    # Upload de fichier
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Afficher l'image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Bouton pour lancer la prédiction
        if st.sidebar.button("Classify"):
            with st.spinner("Analyzing image..."):
                try:
                    # Préparer l'image
                    img_array = read_image(uploaded_file)
                    
                    # Faire la prédiction
                    class_prediction = model.predict(img_array)
                    predicted_class = np.argmax(class_prediction[0])
                    class_names = ["apple", "banana", "orange"]
                    predicted_class_name = class_names[predicted_class]
                    confidence = float(class_prediction[0][predicted_class])

                    # Afficher les résultats
                    st.success(f"Prediction: {predicted_class_name.title()}")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()