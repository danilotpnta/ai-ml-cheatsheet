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
    # Custom CSS to improve responsiveness
    # Custom CSS to improve responsiveness
    st.markdown(
        """
        <style>
        /* Default three columns layout */
        [data-testid="column"] {
            width: calc(33.3333% - 1rem) !important;
            flex: 1 1 calc(33.3333% - 1rem) !important;
            min-width: calc(33.3333% - 1rem) !important;
        }

        /* Two columns on medium screens */
        @media (max-width: 1200px) {
            [data-testid="column"] {
                width: calc(50% - 1rem) !important;
                flex: 1 1 calc(50% - 1rem) !important;
                min-width: calc(50% - 1rem) !important;
            }
        }

        /* Stack columns on small screens */
        @media (max-width: 900px) {
            [data-testid="column"] {
                width: calc(100% - 1rem) !important;
                flex: 1 1 calc(100% - 1rem) !important;
                min-width: calc(100% - 1rem) !important;
            }
        }
        </style>
        """, 
        unsafe_allow_html=True,
    )
   
    # Define columns as usual
    col1, col2, col3 = st.columns(3)

    #######################################
    # COLUMN 1
    #######################################
    
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

    #######################################
    # COLUMN 2
    #######################################
    
    col2.subheader('Training Loop')
    col2.code('''
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    ''')

    col2.subheader('Model Evaluation')
    col2.code('''
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
    # COLUMN 3
    #######################################

    col3.subheader('Data Augmentation')
    col3.code('''
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

    col3.subheader('Model Checkpoints')
    col3.code('''
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

    return None


# Run main()

if __name__ == '__main__':
    main()
