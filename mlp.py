import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the dataset
data = pd.read_csv('cleaned_real_estate_data.csv')
data = data[data['Price (€)'] != "Preis auf Anfrage"]
data.fillna(method='ffill', inplace=True)
data['Property Type'] = data['Property Type'].astype('category').cat.codes
data['Location'] = data['Location'].astype('category').cat.codes
data['Neighborhood'] = data['Neighborhood'].astype('category').cat.codes
data['Price (€)'] = pd.to_numeric(data['Price (€)'].str.replace('.', '').str.replace('€', '').str.strip(), errors='coerce')
data.dropna(subset=['Price (€)'], inplace=True)
scaler = StandardScaler()
data[['Price (€)', 'Surface (m²)']] = scaler.fit_transform(data[['Price (€)', 'Surface (m²)']])

X = data[['Property Type', 'Location', 'Surface (m²)']].values
y = data['Price (€)'].values

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cross-validation function
def cross_validate_mlp(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores, mae_scores, r2_scores = [], [], []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Initialize model, loss function, and optimizer
        model = MLP(input_dim=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            y_pred_tensor = model(X_val_tensor)
            y_pred = y_pred_tensor.squeeze().cpu().tolist()

            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

    # Return average metrics
    return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)

# Perform cross-validation for K = 5, 10, and 20
k_values = [5, 10, 20]
results = {'K': [], 'MSE': [], 'MAE': [], 'R2': []}

for k in k_values:
    mse, mae, r2 = cross_validate_mlp(X, y, k)
    results['K'].append(k)
    results['MSE'].append(mse)
    results['MAE'].append(mae)
    results['R2'].append(r2)
    print(f"K={k}: MSE={mse:.4f}, MAE={mae:.4f}, R^2={r2:.4f}")

# Plot the results
plt.figure(figsize=(12, 4))

# Plot MSE for different K values
plt.subplot(1, 3, 1)
plt.plot(results['K'], results['MSE'], marker='o')
plt.title('MSE vs. K')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')

# Plot MAE for different K values
plt.subplot(1, 3, 2)
plt.plot(results['K'], results['MAE'], marker='o')
plt.title('MAE vs. K')
plt.xlabel('K')
plt.ylabel('Mean Absolute Error')

# Plot R2 for different K values
plt.subplot(1, 3, 3)
plt.plot(results['K'], results['R2'], marker='o')
plt.title('R2 vs. K')
plt.xlabel('K')
plt.ylabel('R^2 Score')

plt.tight_layout()
plt.show()
