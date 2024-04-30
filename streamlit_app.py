import streamlit as st
import numpy as np
import tensorflow
import pandas as pd
from keras.layers import TextVectorization
from keras.saving import load_model

model = load_model('toxic-comment-classification.keras')

df = pd.read_csv("train.csv", sep=",", header = 0)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
list_sentences = df["comment_text"]

max_tokens = 25000
max_len = 100
vectorize = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len,
    standardize="lower_and_strip_punctuation",
)
vectorize.adapt(list_sentences)

st.title("Toxic Comment Classification")
input_text = st.text_area("Enter the comment")

if st.button("Predict"):
    if input_text:
        vector_input_text = vectorize(np.array([input_text]))
        result = model.predict(vector_input_text)
        st.write("The comment is")
        for index, percentage in enumerate(result[0]):
            st.write(f"{list_classes[index]}: {percentage*100:.2f}%")
    else:
        st.write("Please enter a comment.")