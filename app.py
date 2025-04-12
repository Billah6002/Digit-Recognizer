import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import io
import base64

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="wide")

@st.cache_resource
def load_mnist_model():
    try:
        model = load_model('mnist_model.h5')
        return model
    except:
        st.error("Model file not found. Please run train_model.py first.")
        return None

def preprocess_image(image):
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (28, 28))
    
    if np.mean(image) > 127:
        image = 255 - image
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    image = image.reshape(1, 28, 28, 1)
    
    return image

def main():
    model = load_mnist_model()
    
    st.title("✏️ Digit Recognizer")
    st.write("Draw a digit (0-9) in the canvas below and the AI will predict what digit it is!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=28,
            stroke_color="white",
            background_color="black",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas",
        )
    
    with col2:
        if st.button("Predict"):
            if canvas_result.image_data is not None:
                image = canvas_result.image_data
                
                st.write("Processed Image:")
                
                if len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        gray_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                    else:  # RGB
                        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray_image = image.copy()
                
                resized_image = cv2.resize(gray_image, (28, 28))
                
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(resized_image, cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                if model is not None:
                    processed_image = cv2.resize(gray_image, (28, 28))
                    
                    if np.mean(processed_image) > 127:
                        processed_image = 255 - processed_image
                    
                    
                    processed_image = processed_image.astype('float32') / 255.0
                    processed_image = processed_image.reshape(1, 28, 28, 1)
                    
                    prediction = model.predict(processed_image)
                    predicted_digit = np.argmax(prediction)
                    confidence = float(prediction[0][predicted_digit])
                    
                    st.write(f"## Prediction: {predicted_digit}")
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.bar(range(10), prediction[0])
                    ax.set_xticks(range(10))
                    ax.set_xlabel('Digit')
                    ax.set_ylabel('Probability')
                    st.pyplot(fig)
        # clear
        if st.button("Clear Canvas"):
            st.session_state.pop("canvas", None)
            st.rerun()
    
    # instructions
    st.markdown("""
    ### Instructions:
    1. Draw a digit (0-9) in the canvas using your mouse/touchpad
    2. Click 'Predict' to see what digit the AI recognizes
    3. Click 'Clear Canvas' to start over
    
    ### About:
    This application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset, which contains 70,000 images of handwritten digits.
    """)
    
    # additional information
    st.sidebar.title("About this App")
    st.sidebar.info(
        "This application demonstrates how deep learning models can recognize handwritten digits. "
        "The model was trained on the MNIST dataset, which is a popular benchmark in machine learning."
    )
    
    st.sidebar.title("Model Architecture")
    st.sidebar.markdown("""
    - Convolutional Neural Network (CNN)
    - 2 Conv2D layers with ReLU activation
    - MaxPooling and Dropout for regularization
    - Dense layers for classification
    - Trained for 15 epochs with 99% accuracy
    """)

if __name__ == "__main__":
    main()
