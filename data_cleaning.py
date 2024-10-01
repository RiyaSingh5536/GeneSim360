import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('genomic_data.csv')

# Data cleaning
data.drop(columns=['hid', 'hstart', 'hend', 'genscan', 'gene_id', 'transcript_id', 'exon_id', 'gene_type', 'probe_name', 'score'], inplace=True)
data['variation_name'] = data['variation_name'].fillna('unknown')

# Encode categorical features
label_encoder = LabelEncoder()
categorical_columns = ['source', 'feature', 'strand', 'frame', 'variation_name']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Check distribution of the target variable
print("Target variable distribution:")
print(data['variation_name'].value_counts())

# Split the data into features and target
X = data.drop(columns=['variation_name'])  # Features
y = data['variation_name']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model with more complexity
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')  # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_acc}')


model.save('model.keras')


from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    input_data = np.array(data['input']).reshape(1, -1)  # Reshape for a single prediction
    
  
    input_data_scaled = scaler.transform(input_data)
    
   
    prediction = model.predict(input_data_scaled)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

