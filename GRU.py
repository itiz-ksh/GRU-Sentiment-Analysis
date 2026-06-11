import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

# Dataset
df = pd.read_csv('dataset/IMDB Dataset.csv')

# Shape
print(df.head(3))
print(f"\nShape: {df.shape}")
print(f"Label counts:\n{df['sentiment'].value_counts()}")

# Cleaned the data and encoded labels
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['review']    = df['review'].apply(clean_text)
df['sentiment'] = (df['sentiment'] == 'positive').astype(int) # positive -> 1 | negative -> 0

print(df['review'][0][:200])


# Tokenized and paded the dataset
NUM_WORDS = 10000
MAX_LEN   = 200

tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['review'])

sequences = tokenizer.texts_to_sequences(df['review'])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

X = np.array(padded)
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"X_train: {X_train.shape},  X_test: {X_test.shape}")

model = Sequential([
    Embedding(input_dim=NUM_WORDS, output_dim=128, input_length=MAX_LEN),
    Dropout(0.3),
    GRU(64, dropout=0.3, recurrent_dropout=0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training 
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(history.history['accuracy'], label='train')
ax1.plot(history.history['val_accuracy'], label='val')
ax1.set_title('Accuracy'); ax1.legend()

ax2.plot(history.history['loss'], label='train')
ax2.plot(history.history['val_loss'], label='val')
ax2.set_title('Loss'); ax2.legend()

plt.tight_layout(); plt.show()

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc*100:.2f}%")
print(f"Test loss: {loss:.4f}")


# Prediction on model
def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    score = model.predict(padded, verbose=0)[0][0]
    label = "Positive :)" if score > 0.5 else "Negative :("
    print(f"{label}  (score: {score:.3f})")

predict_sentiment("Absolutely brilliant! One of the best films I have seen.")
predict_sentiment("Complete disaster. Dull plot, bad acting, total waste of time.")