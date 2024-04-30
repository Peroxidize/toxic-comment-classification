import streamlit as st
import numpy as np
import re
import tensorflow as tf

def load_model():
    try:
        model = tf.keras.models.load_model('toxic-comment-classification.keras')

        vectorize = tf.keras.layers.TextVectorization(
            max_tokens=25000,
            output_mode='int',
            output_sequence_length=100,
        )
        vectorize.load_assets(".")
        return model, vectorize
    except:
        return load_model


def predict(input_text):
    try:
        vector_input_text = vectorize(np.array([input_text]))
        result = model.predict(vector_input_text)
        st.write("The comment is")
        for index, percentage in enumerate(result[0]):
            st.write(f"{list_classes[index]}: {percentage*100:.2f}%")
    except:
        return predict(input_text)


model, vectorize = load_model()

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.title("Toxic Comment Classification")
input_text = st.text_area("Enter the comment")

if st.button("Predict"):
    if input_text:
        predict(input_text)
    else:
        st.write("Please enter a comment.")