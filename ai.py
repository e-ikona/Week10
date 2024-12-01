import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv('student_dropout.csv')

label_encoder = LabelEncoder()
data['Target'] = label_encoder.fit_transform(data['Target'])

X = data.drop(columns=['Target'])
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def split_data(X, y, train_ratio):
    return train_test_split(X, y, train_size=train_ratio, random_state=42, stratify=y)

configurations = {'50-50': 0.5, '60-40': 0.6, '70-30': 0.7}
splits = {key: split_data(X_scaled, y, ratio) for key, ratio in configurations.items()}

def build_mlp(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

results = {}
for config, (X_train, X_test, y_train, y_test) in splits.items():
    model = build_mlp(X_train.shape[1])
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    results[config] = test_accuracy

print("Test accuracies for each configuration:")
print(results)
