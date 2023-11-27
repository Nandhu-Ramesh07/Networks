import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load word index for Sentiment Classification
word_to_index = imdb.get_word_index()

# Function to perform sentiment classification
def Sentimental_Analysis(new_review_text, model):
    max_review_length = 500
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)
    if type(prediction) == list:
        prediction = prediction[0]
    return "Positive" if prediction > 0.5 else "Negative"

# Function to perform tumor detection
def tumor_detection(img, model):
    img = Image.open(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    return "Tumor Detected" if res else "No Tumor"

# Streamlit App
st.title("Deep Learning")

# Choose between tasks
task = st.selectbox('Select the Task', ['Choose one','Sentimental Analysis', 'Tumor Detection'])

if task == "Sentimental Analysis":
    
    opt= ["Perceptron","DNN", "RNN", "LSTM","Back Propagation"]
    model_option = st.radio("Select", opt, horizontal=True)


    # Load models dynamically based on the selected option
    if model_option == "Perceptron":
        with open('imdb_perceptron.pkl', 'rb') as file:
            model = pickle.load(file)
    elif model_option == "Back Propagation":
        with open('imdb_back_prop.pkl', 'rb') as file:
            model = pickle.load(file)
    elif model_option == "DNN":
        model = load_model('DNN.keras')
    elif model_option == "RNN":
        model = load_model('RNN.keras')
    elif model_option == "LSTM":
        model = load_model('LSTM.keras')
    
    # Input box for new review
    new_review_text = st.text_area("Enter a New Review:", value="")


    if st.button("Predict")and not new_review_text.strip():
        st.warning("Please enter a review.")
    else:
        result = Sentimental_Analysis(new_review_text, model)
        st.subheader("Sentiment Classification Result")
        st.write(f"**{result}**")



elif task == "Tumor Detection":
    st.subheader("Tumor Detection")
    uploaded_file = st.file_uploader("Choose a tumor image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the tumor detection model
        model = load_model('CNN.keras')
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=False, width=200)
        st.write("")

        if st.button("Detect Tumor"):
            result = tumor_detection(uploaded_file, model)
            st.subheader("Tumor Detection Result")
            st.write(f"**{result}**")
