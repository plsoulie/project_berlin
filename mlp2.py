import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 1. Load and preprocess the data
try:
    df = pd.read_csv('cleaned_real_estate_data.csv')
except FileNotFoundError:
    raise FileNotFoundError("Data file 'cleaned_real_estate_data.csv' not found")
except Exception as e:
    raise Exception(f"Error loading data: {str(e)}")

# Handle 'Preis auf Anfrage' and currency symbol in Price - Updated to scale prices to thousands
df['Price (€)'] = df['Price (€)'].replace('Preis auf Anfrage', np.nan)
df['Price (€)'] = df['Price (€)'].str.replace('\xa0€', '').str.replace(' €', '').str.replace('.', '').astype(float)
df['Price (€)'] = df['Price (€)'] / 1000  # Convert to thousands of euros

# Log transform the price values to handle the large scale better
df['Price (€)'] = np.log1p(df['Price (€)'])

# Remove outliers (optional but recommended)
def remove_outliers(df, column, n_std=3):
    mean = df[column].mean()
    std = df[column].std()
    df = df[(df[column] <= mean + (n_std * std)) & 
            (df[column] >= mean - (n_std * std))]
    return df

# Remove outliers from price and surface
df = remove_outliers(df, 'Price (€)')
df = remove_outliers(df, 'Surface (m²)')

# Handle missing values
features = ['Surface (m²)', 'Rooms', 'Floor', 'Neighborhood']
for column in features:
    if df[column].dtype == object:
        df[column] = df[column].replace(['N/A', 'Unknown'], np.nan)
    # Only convert to numeric if it's not the Neighborhood column
    if column != 'Neighborhood':
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column] = df[column].fillna(df[column].median() if column != 'Neighborhood' else df[column].mode()[0])

# Select features for the model
features = ['Surface (m²)', 'Rooms', 'Floor', 'Neighborhood']
target = 'Price (€)'

# Prepare X and y
X = df[features].copy()
y = df[target]

# Encode Zip Code
le = LabelEncoder()
X['Neighborhood'] = le.fit_transform(X['Neighborhood'])

# After the initial feature preparation but before scaling, add:
def add_interaction_features(X):
    """Add interaction terms between surface area and location with improved handling"""
    X = X.copy()
    
    # Create interaction between Surface and Neighborhood
    X['Surface_Neighborhood'] = X['Surface (m²)'] * X['Neighborhood']
    
    # Improved binning logic with error handling
    try:
        if len(X) == 1:
            X['Neighborhood_small'] = 0
            X['Neighborhood_medium'] = 1
            X['Neighborhood_large'] = 0
        else:
            # Use robust quantile binning
            surface_bins = pd.qcut(X['Surface (m²)'], q=3, labels=['small', 'medium', 'large'], 
                                 duplicates='drop', retbins=True)
            bin_edges = surface_bins[1]  # Store bin edges for prediction
            surface_categories = surface_bins[0]
            
            for size in ['small', 'medium', 'large']:
                X[f'Neighborhood_{size}'] = (surface_categories == size).astype(int) * X['Neighborhood']
    except Exception as e:
        raise Exception(f"Error creating interaction features: {str(e)}")
    
    return X

# After preparing X and y but before splitting:
X = add_interaction_features(X)

# Update features list to include new interaction terms
features = ['Surface (m²)', 'Rooms', 'Floor', 'Neighborhood', 
           'Surface_Neighborhood', 'Neighborhood_small', 'Neighborhood_medium', 'Neighborhood_large']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the neural network with modified architecture
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(len(features),)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1)
])

# Compile with a custom learning rate schedule
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9
learning_rate_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate
)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)
model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Add model checkpointing
checkpoint_path = "model_checkpoints/best_model.h5"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\nTest Mean Absolute Error: €{test_mae:,.2f}')

# Function to make predictions
def predict_price(surface, rooms, floor, neighborhood):
    """Enhanced prediction function with input validation"""
    # Input validation
    if not isinstance(surface, (int, float)) or surface <= 0:
        raise ValueError("Surface must be a positive number")
    if not isinstance(rooms, (int, float)) or rooms <= 0:
        raise ValueError("Rooms must be a positive number")
    if not isinstance(floor, (int, float)):
        raise ValueError("Floor must be a number")
    if not isinstance(neighborhood, str):
        raise ValueError("Neighborhood must be a string")
    
    # Create input data
    input_data = pd.DataFrame([[surface, rooms, floor, neighborhood]], 
                            columns=['Surface (m²)', 'Rooms', 'Floor', 'Neighborhood'])
    
    try:
        # Transform neighborhood
        input_data['Neighborhood'] = le.transform([neighborhood])
    except ValueError:
        available_neighborhoods = ", ".join(sorted(le.classes_))
        raise ValueError(f"Invalid neighborhood. Available options are: {available_neighborhoods}")
    
    try:
        # Add interaction features
        input_data = add_interaction_features(input_data)
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = np.expm1(model.predict(input_scaled)[0][0]) * 1000
        return round(prediction, 2)  # Round to 2 decimal places
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

# Example prediction
example_prediction = predict_price(
    surface=80,
    rooms=3,
    floor=2,
    neighborhood='Mitte'
)
print(f'\nPredicted price for example property: €{example_prediction:,.2f}')

# After model training, add these plotting functions:

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE plot
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_scatter(model, X_test_scaled, y_test):
    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)
    
    # Convert back from log scale
    y_test_original = np.expm1(y_test) * 1000
    y_pred_original = np.expm1(y_pred) * 1000
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_original, y_pred_original, alpha=0.5)
    plt.plot([y_test_original.min(), y_test_original.max()], 
             [y_test_original.min(), y_test_original.max()], 
             'r--', lw=2)
    
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title('Predicted vs Actual House Prices')
    plt.grid(True)
    
    # Add R² score
    r2 = r2_score(y_test_original, y_pred_original)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_residuals(model, X_test_scaled, y_test):
    # Make predictions and calculate residuals
    y_pred = model.predict(X_test_scaled).flatten()
    
    # Convert back from log scale
    y_test_original = np.expm1(y_test) * 1000
    y_pred_original = np.expm1(y_pred) * 1000
    
    residuals = y_test_original - y_pred_original
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred_original, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Price (€)')
    ax1.set_ylabel('Residuals (€)')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True)
    
    # Residuals distribution
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residuals (€)')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """Analyze feature importance using a simple sensitivity analysis"""
    baseline = np.zeros((1, len(feature_names)))
    importance = []
    
    for i in range(len(feature_names)):
        temp = baseline.copy()
        temp[0, i] = 1
        importance.append(np.abs(model.predict(temp)[0][0]))
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.show()

def plot_price_by_neighborhood(df):
    """Box plot of prices by neighborhood"""
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='Neighborhood', y='Price (€)', data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Price Distribution by Neighborhood')
    plt.tight_layout()
    plt.show()

def plot_price_surface_relationship(df):
    """Scatter plot with regression line showing price vs surface area"""
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Surface (m²)', y='Price (€)', data=df, scatter_kws={'alpha':0.5})
    plt.title('Price vs Surface Area')
    plt.tight_layout()
    plt.show()

def plot_price_distribution(df):
    """Distribution of prices with KDE"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Price (€)', kde=True)
    plt.title('Distribution of Property Prices')
    plt.xlabel('Price (€)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Correlation matrix of numerical features"""
    numerical_cols = ['Price (€)', 'Surface (m²)', 'Rooms', 'Floor']
    corr = df[numerical_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Add these lines after model training
plot_training_history(history)
plot_prediction_scatter(model, X_test_scaled, y_test)
plot_residuals(model, X_test_scaled, y_test)
plot_feature_importance(model, features)
plot_price_by_neighborhood(df)
plot_price_surface_relationship(df)
plot_price_distribution(df)
plot_correlation_matrix(df)

# After creating the LabelEncoder
print("Available neighborhoods:", list(le.classes_))
