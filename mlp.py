import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('cleaned_real_estate_data.csv')

# Filter out rows where Price is "Preis auf Anfrage"
data = data[data['Price (€)'] != "Preis auf Anfrage"]

# Preprocess the data
data.fillna(method='ffill', inplace=True)
data['Property Type'] = data['Property Type'].astype('category').cat.codes
data['Location'] = data['Location'].astype('category').cat.codes
data['Neighborhood'] = data['Neighborhood'].astype('category').cat.codes

# Convert Price to numeric, forcing errors to NaN, then drop NaN values
data['Price (€)'] = pd.to_numeric(data['Price (€)'].str.replace('.', '').str.replace('€', '').str.strip(), errors='coerce')
data.dropna(subset=['Price (€)'], inplace=True)  # Drop rows with NaN prices

# Normalize numerical features
scaler = StandardScaler()
data[['Price (€)', 'Surface (m²)']] = scaler.fit_transform(data[['Price (€)', 'Surface (m²)']])  # Only normalize Price and Surface

# Define features and target variable, ignoring Rooms and Floor
X = data[['Property Type', 'Location', 'Surface (m²)']].values  # Exclude Rooms and Floor
y = data['Price (€)'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 100)  # Input layer to hidden layer
        self.fc2 = nn.Linear(100, 50)                 # Hidden layer to hidden layer
        self.fc3 = nn.Linear(50, 1)                   # Hidden layer to output layer
        self.relu = nn.ReLU()                          # Activation function

    def forward(self, x):
        x = self.relu(self.fc1(x))                    # First layer
        x = self.relu(self.fc2(x))                    # Second layer
        x = self.fc3(x)                               # Output layer
        return x

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()                          # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training the model
num_epochs = 1000
loss_values = []  # Store training loss values
val_loss_values = []  # Store validation loss values

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()                          # Zero the gradients
    outputs = model(X_train_tensor)                # Forward pass
    loss = criterion(outputs.squeeze(), y_train_tensor)  # Calculate loss
    loss.backward()                                # Backward pass
    optimizer.step()                               # Update weights
    loss_values.append(loss.item())

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs.squeeze(), y_test_tensor)
        val_loss_values.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:                     # Print every 100 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)           # Make predictions
    y_pred = y_pred_tensor.squeeze().cpu().tolist()  # Convert to list instead of numpy array
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error on Test Set: {mse:.4f}')
    print(f'Mean Absolute Error on Test Set: {mae:.4f}')
    print(f'R^2 Score on Test Set: {r2:.4f}')


# Plot learning curves
plt.plot(loss_values, label='Training Loss')
plt.plot(val_loss_values, label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Scatter plot of predicted vs actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.title('Predicted vs Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Feature importance - analyzing first layer weights
weights = model.fc1.weight.detach().cpu().tolist()  # Convert to list instead of numpy array
importance = [sum(abs(w) for w in weight) / len(weight) for weight in zip(*weights)]  # Calculate mean absolute importance per feature
print("Feature Importance based on First Layer Weights:", importance)
