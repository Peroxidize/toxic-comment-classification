import streamlit as st
import numpy as np
import re
import tensorflow as tf

model = tf.keras.models.load_model('toxic-comment-classification.keras')

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=25000,
    output_mode='int',
    output_sequence_length=100,
)
vectorize.load_assets(".")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.title("Toxic Comment Classification")
input_text = st.text_area("Enter the comment")
temp_text = ""

if st.button("Predict"):
    if input_text:
        temp_text = input_text
        try:
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text)
            vector_input_text = vectorize(np.array([cleaned_text]))
            result = model.predict(vector_input_text)
            st.write("The comment is")
            for index, percentage in enumerate(result[0]):
                st.write(f"{list_classes[index]}: {percentage*100:.2f}%")
        except:
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', temp_text)
            vector_input_text = vectorize(np.array([cleaned_text]))
            result = model.predict(vector_input_text)
            st.write("The comment is")
            for index, percentage in enumerate(result[0]):
                st.write(f"{list_classes[index]}: {percentage*100:.2f}%")
    else:
        st.write("Please enter a comment.")