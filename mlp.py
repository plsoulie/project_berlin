import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('cleaned_real_estate_data.csv')

# Preprocess the data
data.fillna(method='ffill', inplace=True)
data['Property Type'] = data['Property Type'].astype('category').cat.codes
data['Location'] = data['Location'].astype('category').cat.codes
data['Neighborhood'] = data['Neighborhood'].astype('category').cat.codes

# Normalize numerical features
scaler = StandardScaler()
data[['Price (€)', 'Surface (m²)', 'Rooms', 'Floor']] = scaler.fit_transform(data[['Price (€)', 'Surface (m²)', 'Rooms', 'Floor']])

# Define features and target variable
X = data[['Property Type', 'Location', 'Surface (m²)', 'Rooms', 'Floor']].values
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
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()                          # Zero the gradients
    outputs = model(X_train_tensor)                # Forward pass
    loss = criterion(outputs.squeeze(), y_train_tensor)  # Calculate loss
    loss.backward()                                # Backward pass
    optimizer.step()                               # Update weights

    if (epoch + 1) % 100 == 0:                     # Print every 100 epochs
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)           # Make predictions
    mse = criterion(y_pred_tensor.squeeze(), y_test_tensor)  # Calculate MSE
    print(f'Mean Squared Error on Test Set: {mse.item():.4f}')
