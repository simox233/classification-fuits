import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import time
import json
from datetime import datetime
import base64

# Configuration de la page
st.set_page_config(
    page_title="🍎 AI Fruit Analytics",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS avancés
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700&display=swap');

    * {
        font-family: 'Roboto', sans-serif;
    }

    .main {
        background-color: #f8f9fa;
    }

    .stButton>button {
        width: 100%;
        padding: 15px 0;
        border: none;
        border-radius: 15px;
        background: linear-gradient(45deg, #FF512F 0%, #F09819 100%);
        color: white;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    .card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    .metric-card {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 15px;
        margin: 10px 0;
    }

    .prediction-label {
        font-size: 24px;
        font-weight: 700;
        color: #1e3c72;
        text-align: center;
        margin: 20px 0;
    }

    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    }

    .history-item {
        padding: 10px;
        border-left: 4px solid #1e3c72;
        margin: 10px 0;
        background-color: #f8f9fa;
        border-radius: 0 10px 10px 0;
    }

    .sidebar-info {
        padding: 15px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
def load_history():
    try:
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    except:
        return []

def save_history(history):
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f)

def read_image(uploaded_file):
    temp_file = f"temp_image.{uploaded_file.name.split('.')[-1]}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    img = Image.open(temp_file).convert("RGB")
    img = img.resize((32, 32))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    #x = x / 255.0
    
    os.remove(temp_file)
    return x, img

def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="predicted_image.jpg">Download result</a>'
    return href

class_names = ["Apple 🍎", "Banana 🍌", "Orange 🍊"]
class_descriptions = {
    "Apple 🍎": "Rich in fiber and antioxidants, apples are one of the most popular fruits worldwide.",
    "Banana 🍌": "High in potassium and natural sugars, bananas are perfect for quick energy.",
    "Orange 🍊": "Excellent source of vitamin C and immune system boosting compounds."
}

def main():
    # Sidebar
    st.sidebar.markdown("""
        <div style='text-align: center;'>
            <h2>🎯 AI Fruit Analytics</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Menu de navigation
    page = st.sidebar.selectbox("Navigation", ["Classifier", "History", "About"])

    if page == "Classifier":
        show_classifier_page()
    elif page == "History":
        show_history_page()
    else:
        show_about_page()

def show_classifier_page():
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("""
            <div class="card">
                <h3>📸 Upload Image</h3>
                <p>Support formats: JPG, PNG</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        # Charger le modèle
        @st.cache_resource
        def load_classifier():
            return load_model('model.h5')
        
        try:
            model = load_classifier()
        except Exception as e:
            st.error("⚠️ Model loading failed!")
            st.stop()

    with col2:
        if uploaded_file:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🔮 Analyze Image"):
                with st.spinner("🎯 Analyzing your image..."):
                    try:
                        # Progress animation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            status_text.text(f"Processing... {i+1}%")
                            time.sleep(0.01)

                        # Prediction
                        img_array, original_img = read_image(uploaded_file)
                        class_prediction = model.predict(img_array)
                        predicted_class = np.argmax(class_prediction[0])
                        confidence = float(class_prediction[0][predicted_class])

                        # Affichage des résultats
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                                <div class="metric-card">
                                    <h4>Prediction</h4>
                                    <h2>{}</h2>
                                </div>
                            """.format(class_names[predicted_class]), unsafe_allow_html=True)

                        with col2:
                            st.markdown("""
                                <div class="metric-card">
                                    <h4>Confidence</h4>
                                    <h2>{:.1f}%</h2>
                                </div>
                            """.format(confidence * 100), unsafe_allow_html=True)

                        with col3:
                            st.markdown("""
                                <div class="metric-card">
                                    <h4>Processing Time</h4>
                                    <h2>1.0s</h2>
                                </div>
                            """.format(confidence * 100), unsafe_allow_html=True)

                        # Detailed Analysis
                        st.markdown("""
                            <div class="card">
                                <h3>📊 Detailed Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        for name, prob in zip(class_names, class_prediction[0]):
                            st.markdown(f"""
                                <div style='margin: 10px 0;'>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span>{name}</span>
                                        <span>{prob:.1%}</span>
                                    </div>
                                    <div class="confidence-bar" style='width: {prob:.1%}'></div>
                                </div>
                            """, unsafe_allow_html=True)

                        # Fruit Information
                        st.markdown("""
                            <div class="card">
                                <h3>ℹ️ Fruit Information</h3>
                                <p>{}</p>
                            </div>
                        """.format(class_descriptions[class_names[predicted_class]]), unsafe_allow_html=True)

                        # Save prediction to history
                        history = load_history()
                        history.append({
                            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'prediction': class_names[predicted_class],
                            'confidence': float(confidence),
                        })
                        save_history(history)

                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.markdown("""
                <div class="card" style='text-align: center;'>
                    <h2>👋 Welcome to AI Fruit Analytics!</h2>
                    <p>Upload an image to start the analysis</p>
                    <div style='margin: 20px 0;'>
                        <h4>Supported Fruits:</h4>
                        <p>🍎 Apple  •  🍌 Banana  •  🍊 Orange</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

def show_history_page():
    st.markdown("<h2>📚 Prediction History</h2>", unsafe_allow_html=True)
    
    history = load_history()
    if not history:
        st.info("No predictions yet! Start by analyzing some images.")
        return

    # Statistics
    total_predictions = len(history)
    avg_confidence = sum(h['confidence'] for h in history) / total_predictions
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h4>Total Predictions</h4>
                <h2>{}</h2>
            </div>
        """.format(total_predictions), unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4>Average Confidence</h4>
                <h2>{:.1f}%</h2>
            </div>
        """.format(avg_confidence * 100), unsafe_allow_html=True)

    # History list
    st.markdown("<h3>Recent Predictions</h3>", unsafe_allow_html=True)
    for item in reversed(history[-10:]):  # Show last 10 predictions
        st.markdown(f"""
            <div class="history-item">
                <strong>{item['prediction']}</strong> • 
                Confidence: {item['confidence']:.1%} • 
                Date: {item['date']}
            </div>
        """, unsafe_allow_html=True)

def show_about_page():
    st.markdown("""
        <div class="card">
            <h2>🤖 About AI Fruit Analytics</h2>
            <p>AI Fruit Analytics is a state-of-the-art fruit classification system powered by deep learning. 
            Our model is trained on thousands of images to accurately identify different types of fruits.</p>
            
            <h3>🎯 Features</h3>
            <ul>
                <li>Real-time fruit classification</li>
                <li>High accuracy predictions</li>
                <li>Detailed analysis with confidence scores</li>
                <li>Historical tracking of predictions</li>
            </ul>
            
            <h3>🍎 Supported Fruits</h3>
            <ul>
                <li>Apples</li>
                <li>Bananas</li>
                <li>Oranges</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
