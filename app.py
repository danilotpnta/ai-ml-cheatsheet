"""
AI/ML Development Cheat Sheet

App to summarise common AI/ML development practices

v1.0.0
2 September 2024

Author:
    Your Name : https://github.com/danilotpnta

"""

import streamlit as st
from pathlib import Path
import base64

# Initial page config

st.set_page_config(
     page_title='AI/ML Cheat Sheet',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    cs_sidebar()
    cs_body()

    return None

# Convert image to bytes (if needed)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# Sidebar

def cs_sidebar():

    st.sidebar.header('AI/ML Cheat Sheet')
    st.sidebar.markdown('''
<small>Summary of common practices in AI/ML development.</small>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown('__Setup Python Environment__')

    st.sidebar.code('''
# Create a new conda environment
$ conda create -n myenv python=3.8

# Activate the environment
$ conda activate myenv

# Install packages
$ pip install numpy pandas torch transformers
    ''')

    st.sidebar.markdown('__Set up Requirements File__')
    st.sidebar.code('''
# Create a requirements.txt file
numpy==1.21.0
pandas==1.3.0
torch==1.10.0
transformers==4.9.0

# Install from requirements.txt
$ pip install -r requirements.txt
    ''')

    st.sidebar.markdown('__Create a README__')
    st.sidebar.code('''
# Create a README.md file
# Example template:

# Project Title
A brief description of the project.

## Installation
Instructions to set up the environment.

## Usage
How to run the project.

## Contributing
Guidelines for contributing to the project.
    ''')

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Cheat sheet v1.0.0](https://github.com/danilotpnta/ai-ml-cheatsheet)  | Sep 2024 | [Danilo Toapanta](https://danilotpnta.com)</small>''', unsafe_allow_html=True)

    return None

##########################
# Main body of cheat sheet
##########################

def cs_body():

    col1, col2, col3 = st.columns(3)

    #######################################
    # COLUMN 1
    #######################################
    
    # Display text

    col1.subheader('Preprocessing & Data Handling')
    col1.code('''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Normalize data
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized_data = (data - mean) / std

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Handle missing data
df.fillna(df.mean(), inplace=True)
    ''')

    # Model setup

    col1.subheader('Model Setup')
    col1.code('''
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()

# Choose loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    ''')

    # Training loop

    col1.subheader('Training Loop')
    col1.code('''
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    ''')

    # Evaluation

    col1.subheader('Model Evaluation')
    col1.code('''
# Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')
    ''')

    #######################################
    # COLUMN 2
    #######################################

    # Data augmentation

    col2.subheader('Data Augmentation')
    col2.code('''
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Apply transformations to dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    ''')

    # Checkpoints

    col2.subheader('Model Checkpoints')
    col2.code('''
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ''')

    # Hyperparameter tuning

    col2.subheader('Hyperparameter Tuning')
    col2.code('''
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'batch_size': [16, 32, 64],
    'lr': [0.001, 0.01, 0.1],
    'dropout_rate': [0.3, 0.5, 0.7],
}

# Initialize and run grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
    ''')

    # Experiment tracking

    col2.subheader('Experiment Tracking')
    col2.code('''
import wandb

# Initialize WandB
wandb.init(project="my-ai-project")

# Log hyperparameters and metrics
wandb.config.update({
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32
})

# During training
for epoch in range(num_epochs):
    wandb.log({"epoch": epoch, "loss": loss.item(), "accuracy": accuracy})

wandb.finish()
    ''')

    #######################################
    # COLUMN 3
    #######################################

    # Post-processing

    col3.subheader('Post-processing')
    col3.code('''
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Get predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{cm}')

# Classification report
cr = classification_report(y_test, y_pred)
print(f'Classification Report: \n{cr}')
    ''')

    # Inference

    col3.subheader('Inference')
    col3.code('''
# Switch model to evaluation mode
model.eval()

# Make predictions on new data
with torch.no_grad():
    predictions = model(new_data)

# Apply softmax to get probabilities
probabilities = torch.softmax(predictions, dim=1)
predicted_labels = torch.argmax(probabilities, dim=1)
print(f'Predicted labels: {predicted_labels}')
    ''')

    # Deployment

    col3.subheader('Model Deployment')
    col3.code('''
import joblib

# Save the model
joblib.dump(model, 'model.pkl')

# Load the model
loaded_model = joblib.load('model.pkl')

# Predict using the loaded model
predictions = loaded_model.predict(X_new)
print(f'Predictions: {predictions}')
    ''')

    # Common Mistakes and Tips

    col3.subheader('Common Mistakes & Tips')
    col3.code('''
# Don’t forget to zero gradients
optimizer.zero_grad()

# Always normalize your data
normalized_data = (data - mean) / std

# Use batches to handle large datasets
for batch in dataloader:
    ...

# Track your experiments for better reproducibility
wandb.init(project="my-ai-project")
    ''')

    return None

# Run main()

if __name__ == '__main__':
    main()
