import streamlit as st
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model(
    "models/sentiment_lstm.keras"
)

# Load tokenizer
with open(
    "models/tokenizer.pkl",
    "rb"
) as f:
    tokenizer = pickle.load(f)

st.title(
    "😊 Twitter Sentiment Analysis using LSTM"
)

tweet = st.text_area(
    "Enter Tweet"
)

if st.button(
    "Analyze Sentiment"
):

    sequence = tokenizer.texts_to_sequences(
        [tweet]
    )

    padded = pad_sequences(
        sequence,
        maxlen=50
    )

    prediction = model.predict(
        padded
    )[0][0]

    if prediction >= 0.5:

        st.success(
            f"Positive Sentiment ({prediction:.2f})"
        )

    else:

        st.error(
            f"Negative Sentiment ({prediction:.2f})"
        )