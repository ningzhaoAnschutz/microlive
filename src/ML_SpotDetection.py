import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

from torchvision import transforms
from torchvision.transforms import functional as TF
import random

RESHAPE_IMAGE_SIZE = 64
perform_validation = False

def random_rotate_image(image):
    k = random.choice([0, 1, 2, 3])  # Randomly choose 1, 2, or 3
    if k > 0:
        rotated_image = np.rot90(image, k)
    else:
        rotated_image = image
    return rotated_image



def load_model(model, path):
    # pick target device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # always map to CPU or CUDA
        state_dict = torch.load(path, map_location=device)
    except Exception as e:
        #print("Warning loading model on", device, "– retrying on CPU:", e)
        state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #print(f"Model loaded to {device}.")



def normalize_crop_return_list(array_crops_YXC,crop_size,selected_color_channel=0, normalize_to_255 = False):
    list_crops = []
    number_crops = array_crops_YXC.shape[0] // crop_size
    for crop_id in range(number_crops):
        crop = array_crops_YXC[crop_id * crop_size:(crop_id + 1) * crop_size, :, selected_color_channel]
        crop = crop - np.percentile(crop, 0.01)
        crop = crop / np.percentile(crop, 99.95)
        if normalize_to_255:
            crop = np.clip(crop, 0, 1)
            crop = (crop * 255).astype(np.uint8)
        else:
            crop = np.clip(crop, 0, 1)
        list_crops.append(crop)
    return list_crops

def standardize_spot_return_list(array_crops_YXC, crop_size, selected_color_channel=0):
    list_crops = []
    number_crops = array_crops_YXC.shape[0] // crop_size
    for crop_id in range(number_crops):
        crop = array_crops_YXC[crop_id * crop_size:(crop_id + 1) * crop_size, :, selected_color_channel]
        crop = (crop - np.mean(crop)) / np.std(crop)
        list_crops.append(crop)
    return list_crops

def standarize_crop(crop):
    return (crop - np.mean(crop)) / np.std(crop)

def normalize_crop(crop):
    crop= ((crop - np.min(crop)) / (np.max(crop) - np.min(crop))) #* 255
    return crop

def predict_crops(model, list_crops,threshold=0.5):
    model.eval()
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #model.to(device)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    flag_vector = []
    for crop in list_crops:
        # normalize the original image from 255 to 0-1
        crop = np.array(Image.fromarray(crop).resize((RESHAPE_IMAGE_SIZE, RESHAPE_IMAGE_SIZE))).astype(np.float32)  / 255.0
        crop_tensor = torch.tensor(crop).unsqueeze(0).unsqueeze(0).to(device)  # Move input to the same device as the model
        with torch.no_grad():  # Disable gradient computation
            output = model(crop_tensor)
            prediction = (torch.sigmoid(output) > threshold).float().item()  # Convert output to label (0 or 1)
        flag_vector.append(int(prediction))
    return np.array(flag_vector)

def save_model(model, path='particle_detection_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model state dictionary saved to {path}")

class ParticleDetectionCNN(nn.Module):
    def __init__(self):
        super(ParticleDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=RESHAPE_IMAGE_SIZE, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(RESHAPE_IMAGE_SIZE * 16 * 16, 128)  # This will be updated dynamically
        self.fc2 = nn.Linear(128, 1)  # Binary classification (particle or not)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Dynamically calculate the number of features for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor, keeping batch size
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x

    
class ParticleDataset(Dataset):
    def __init__(self, images_dir, subset='train', use_transform=False):
        self.images_dir = images_dir
        #self.transform = transform
        self.use_transform = use_transform
        images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.png')]
        labels = [0 if 'no_particle' in img else 1 for img in images]
        # Shuffling and splitting data
        combined = list(zip(images, labels))
        random.seed(42)
        random.shuffle(combined)
        images, labels = zip(*combined)
        total_images = len(images)
        train_end = int(total_images * 0.8)
        valid_end = int(total_images * 1)
        #print(f"Total images: {total_images}, Train images: {train_end}, Validation images: {valid_end - train_end}, Test images: {total_images - valid_end}")
        if subset == 'train':
            self.images = images[:train_end]
            self.labels = labels[:train_end]
        elif subset == 'valid':
            self.images = images[train_end:valid_end]
            self.labels = labels[train_end:valid_end]
        elif subset == 'test':
            self.images = images[valid_end:]
            self.labels = labels[valid_end:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        image = image.resize((RESHAPE_IMAGE_SIZE, RESHAPE_IMAGE_SIZE))
        # randomly add a rotation to the image of 90, 180, or 270 degrees
        if self.use_transform:
            image = random_rotate_image(image)
        # normalize image to [0, 1] 
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image to range [0, 1]
        image = torch.tensor(image).unsqueeze(0)  # Convert to tensor and add channel dimension
        
        return image, label


def validate(model, loader, criterion, device):
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            #inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float().to(device)  # Move inputs to the GPU
            labels = labels.unsqueeze(1).float().to(device)  # Move labels to the GPU
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
    return validation_loss / len(loader)


def run_network(image_dir='training_crops', num_epochs=10000, learning_rate=0.0000005, batch_size=256, perform_validation=perform_validation):
    
    train_dataset = ParticleDataset(image_dir, subset='train', use_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if perform_validation:
        valid_dataset = ParticleDataset(image_dir, subset='valid', use_transform=False)  # No augmentation for validation
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # adapt to windows or mac
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = ParticleDetectionCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)  # Move inputs to the GPU
            labels = labels.unsqueeze(1).float().to(device)  # Move labels to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        training_losses.append(running_loss / len(train_loader))
        if perform_validation:
            validation_loss = validate(model, valid_loader, criterion, device)
        else:
            validation_loss = 0
        validation_losses.append(validation_loss)
        if (epoch == 0) or ((epoch ) % batch_size == 0):
            print(f"Epoch {epoch}/{num_epochs} , Training Loss: {running_loss / len(train_loader)}, Validation Loss: {validation_loss}")

    return model, training_losses, validation_losses



