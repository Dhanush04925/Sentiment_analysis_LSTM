import pandas as pd
import numpy as np
import pickle
import re

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ----------------------------------
# Load Dataset
# ----------------------------------

columns = [
    "target",
    "id",
    "date",
    "flag",
    "user",
    "text"
]

df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    names=columns,
    header=None,
    low_memory=False
)

# Keep required columns
df = df[["target", "text"]]

# Convert target to numeric
df["target"] = pd.to_numeric(
    df["target"],
    errors="coerce"
)

# Remove invalid rows
df = df.dropna()

# Convert datatype
df["target"] = df["target"].astype(int)

# Convert labels
# 0 = Negative
# 4 = Positive → 1
df["target"] = df["target"].replace({
    4: 1
})

print(df.head())
print(df.dtypes)

# ----------------------------------
# Reduce Dataset Size (Optional)
# ----------------------------------

df = df.sample(
    n=50000,
    random_state=42
)

# ----------------------------------
# Text Cleaning
# ----------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["text"] = df["text"].apply(clean_text)

# ----------------------------------
# Features & Labels
# ----------------------------------

X = df["text"]

y = df["target"].astype(np.int32)

# ----------------------------------
# Tokenization
# ----------------------------------

max_words = 10000

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X)

with open(
    "models/tokenizer.pkl",
    "wb"
) as f:
    pickle.dump(tokenizer, f)

# Convert text to sequences
X_seq = tokenizer.texts_to_sequences(X)

max_len = 50

X_pad = pad_sequences(
    X_seq,
    maxlen=max_len,
    padding="post",
    truncating="post"
)

X_pad = np.array(
    X_pad,
    dtype=np.int32
)

# ----------------------------------
# Train/Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.2,
    random_state=42
)

# Convert labels to numpy array
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)

print("X dtype:", X_train.dtype)
print("y dtype:", y_train.dtype)

# ----------------------------------
# Build LSTM Model
# ----------------------------------

model = Sequential()

model.add(
    Embedding(
        input_dim=max_words,
        output_dim=128
    )
)

model.add(
    LSTM(
        units=64
    )
)

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Build model before summary
model.build(
    input_shape=(None, max_len)
)

model.summary()

# ----------------------------------
# Train
# ----------------------------------

model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# ----------------------------------
# Save Model
# ----------------------------------

model.save(
    "models/sentiment_lstm.keras"
)

print("✅ Model Saved Successfully")