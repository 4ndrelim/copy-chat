# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocessing function
def preprocess_data(df):
    """Cleans and preprocesses text and labels in a DataFrame."""
    # Drop rows where 'text' or 'sentiment' is NaN
    df = df.dropna(subset=['text', 'sentiment']).copy()

    # Remove extra spaces, lowercase all text
    df.loc[:, 'text'] = df['text'].astype(str).str.strip().str.lower()

    # Remove unwanted characters
    df.loc[:, 'text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))

    # Standardize sentiment labels
    df.loc[:, 'sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()

    # Keep only valid sentiment labels
    valid_labels = {'positive', 'negative', 'neutral'}
    df = df[df['sentiment'].isin(valid_labels)].copy()

    # Remove duplicate rows (optional)
    df = df.drop_duplicates()

    return df

# Load datasets
train_data = pd.read_csv('dataset/train_vanilla.csv', encoding="latin-1")
test_data = pd.read_csv('dataset/test.csv', encoding="latin-1")

# Apply preprocessing
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['sentiment'])
y_test = label_encoder.transform(test_data['sentiment'])

# Train and Evaluate Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

print("Naive Bayes Results:")
print(f"Accuracy: {accuracy_nb:.4f}, Precision: {precision_nb:.4f}, Recall: {recall_nb:.4f}, F1 Score: {f1_nb:.4f}")
print()

# Train and Evaluate SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

print("SVM Results:")
print(f"Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1 Score: {f1_svm:.4f}")
print()

# Tokenize the text data for LSTM
tokenizer = Tokenizer(num_words=10000)  # Increased vocab size
tokenizer.fit_on_texts(train_data['text'])
X_train_seq = tokenizer.texts_to_sequences(train_data['text'])
X_test_seq = tokenizer.texts_to_sequences(test_data['text'])

# Pad sequences
maxlen = 150  # Increased max length for better context
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=10000, output_dim=128))  # Removed input_length to avoid warning
lstm_model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))  # Increased LSTM units
lstm_model.add(Dense(3, activation='softmax'))  # 3 classes: positive, negative, neutral

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train LSTM model
lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1)

# Predict with LSTM
y_pred_lstm = lstm_model.predict(X_test_pad)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)

# Evaluate LSTM
accuracy_lstm = accuracy_score(y_test, y_pred_lstm)
precision_lstm = precision_score(y_test, y_pred_lstm, average='weighted')
recall_lstm = recall_score(y_test, y_pred_lstm, average='weighted')
f1_lstm = f1_score(y_test, y_pred_lstm, average='weighted')

print("LSTM Results:")
print(f"Accuracy: {accuracy_lstm:.4f}, Precision: {precision_lstm:.4f}, Recall: {recall_lstm:.4f}, F1 Score: {f1_lstm:.4f}")
