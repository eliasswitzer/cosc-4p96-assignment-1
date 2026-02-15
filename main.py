import argparse

import torch
from torchvision import datasets
import torchvision.transforms.v2 as v2
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt

from data_prep import compute_mean_std, compute_min_max, MinMaxScaling
from model_supervised import SupervisedNetwork, init_weights
from train_test import train, test
from semi_supervised import co_train, CoTrainingDataset, split_views, predict_with_both_models

parser = argparse.ArgumentParser()

# TODO: Add defaults in all functions and classes, and add option for None to specify that those aren't being used
# TODO: Add parameters to add or skip parts of the pipeline

# Arguments
parser.add_argument("-s", "--seed", metavar="seed", required=True, help="The random seed for reproducibility.")

# Data Splitting
parser.add_argument("--train_split", metavar="train_split", required=False, default=0.8, help="The percentage of the data to be used for training.")
parser.add_argument("--val_split", metavar="val_split", required=False, default=0.1, help="The percentage of the data to be used for validation.")
parser.add_argument("--labeled_split", metavar="labeled_split", default=0.2, required=False, help="The percentage of the training data to be labeled.")
parser.add_argument("--initial_train_split", metavar="initial_train_split", default=0.75, required=False, help="The percentage of labeled data to train supervised network with.")

# Data Preprocessing and Augmentation
parser.add_argument("--preprocessing_type", metavar="preprocessing_type", required=False, default="zscore", help="The type of preprocessing to use, either zscore or minmax")
parser.add_argument("--scaling_min", metavar="scaling_min", required=False, default=0, help="The minimum value for min-max scaling.")
parser.add_argument("--scaling_max", metavar="scaling_max", required=False, default=1, help="The maximum value for min-max scaling.")
parser.add_argument("--p_horizonal_flip", metavar="p_horizontal_flip", required=False, default=0.5, help="The probability of a horizontal flip.")
parser.add_argument("--std_noise", metavar="std_noise", required=False, default=0.1, help="The standard deviation for the distribution to sample added noise from.")

# Mini-Batches
parser.add_argument("--init_train_batch_size", metavar="init_train_batch_size", required=False, default=64, help="The batch size for initial supervised training.")
parser.add_argument("--init_val_batch_size", metavar="init_val_batch_size", required=False, default=64, help="The batch size for initial supervised validation.")

# Model Parameters
parser.add_argument("-e", "--epochs", metavar="epochs", required=False, default=30, help="The number of epochs for training")
parser.add_argument("-lr", "--learning_rate", metavar="learning_rate", required=False, default=0.01, help="The learning rate of the model, try values in range [0.001, 0.1].")
parser.add_argument("-m", "--momentum", metavar="momentum", required=False, default=0.9, help="The momentum value of the model, try values in range [0.5, 0.99]")
parser.add_argument("--init_type", metavar="init_type", required=False, default="he", help="The weight initialization type, either random_uniform, random_normal, or he.")
parser.add_argument("-ow", "--overfitting_window", metavar="overfitting_window", required=False, default=10, help="The number of previous epochs to check for overfitting.")
parser.add_argument("-l2", "--lambda_l2", metavar="lambda_l2", required=False, default=1e-4, help="Controls the lambda in weight decay regularization")
parser.add_argument("--p_dropout", metavar="p_dropout", required=False, default=0.5, help="Controls the dropout probability")

args = parser.parse_args()

# Set random seeds
torch.manual_seed(int(args.seed))
random.seed(int(args.seed))
np.random.seed(int(args.seed))
generator = torch.Generator().manual_seed(int(args.seed))

# Set device based on what is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # will run on GPU if available
print("Using device: ", device)

# Load CIFAR-10 Dataset and apply data transformations
transform = v2.Compose([v2.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

data_mean, data_std = compute_mean_std(dataset)
data_min, data_max = compute_min_max(dataset)

# Split Data and Apply Augmentation (to train set only)

# This custom split is from the official training data and does not use the official testing data
# Split into training, validation and testing data
train_set, val_set, test_set = random_split(dataset, [float(args.train_split), float(args.val_split), 1-float(args.train_split)-float(args.val_split)], generator=generator)

if args.preprocessing_type == "zscore":
    train_set.transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=data_mean, std=data_std), # Apply z-score normalization

        v2.RandomHorizontalFlip(p=float(args.p_horizonal_flip)),
        v2.RandomCrop((32,32)),
        v2.GaussianNoise(mean=0, sigma=float(args.std_noise)),
        v2.ColorJitter()
    ])
else:
    train_set.transform = v2.Compose([
        v2.ToTensor(),
        MinMaxScaling(data_min, data_max, float(args.scaling_min), float(args.scaling_max)), # Apply min-max scaling to [0,1]

        v2.RandomHorizontalFlip(p=float(args.p_horizonal_flip)),
        v2.RandomCrop((32,32)),
        v2.GaussianNoise(mean=0, sigma=float(args.std_noise)),
        v2.ColorJitter()
    ])

val_set.transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=data_mean, std=data_std), # Apply z-score normalization
])

test_set.transform = v2.Compose([
    v2.ToTensor(),
    v2.Normalize(mean=data_mean, std=data_std), # Apply z-score normalization
])

# Split into labeled and unlabeled for semi-supervised setup
labeled_training_set, unlabeled_training_set = random_split(train_set, [float(args.labeled_split), 1-float(args.labeled_split)], generator=generator)

# Split into initial training and validation (from the labeled data)
initial_training_set, initial_validation_set = random_split(labeled_training_set, [float(args.initial_train_split), 1-float(args.initial_train_split)], generator=generator)

# Visualize Augmented Dataset
labels = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    img, label = dataset[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels[label])
    plt.axis("off")
    img = img.permute(1,2,0)
    plt.imshow(img.squeeze(), cmap="viridis")
plt.show()

# Create Data Loaders
initial_train_loader = DataLoader(initial_training_set, batch_size=int(args.init_train_batch_size), shuffle=True, generator=generator) # specifying batch size > 1 ensures mini-batch gradient descent is used
initial_val_loader = DataLoader(initial_validation_set, batch_size=int(args.init_val_batch_size), generator=generator)
test_loader = DataLoader(test_set, batch_size=64, generator=generator)

# Configure supervised neural network, weight initialization, optimizer and loss function
model = SupervisedNetwork()
model.apply(lambda m: init_weights(m, type=args.init_type, generator=generator))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=float(args.learning_rate), momentum=float(args.momentum))

# Train and validate baseline supervised neural network
metrics = train(model, device, int(args.epochs), initial_train_loader, initial_val_loader, optimizer, criterion, int(args.overfitting_window), float(args.lambda_l2))

# Test model
average_test_loss, test_acc = test(model, device, test_loader, criterion)

# Visualize training and validation loss
epochs = range(1, len(metrics['train_loss']) + 1)
plt.figure()
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualize training and validation accuracy
plt.figure()
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# -----Semi-Supervised Learning: Co-Training-----
# TODO: Test this actually works and add parameters to args

# Initialize model A and model B
model_A = SupervisedNetwork(half_images=True).to(device)
model_B = SupervisedNetwork(half_images=True).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer_A = torch.optim.Adam(model_A.parameters(), lr=0.001)
optimizer_B = torch.optim.Adam(model_B.parameters(), lr=0.001)

all_left = []
all_right = []
all_labels = []
for image, label in labeled_training_set:
    left, right = split_views(image)
    all_left.append(left.unsqueeze(0))
    all_right.append(right.unsqueeze(0))
    all_labels.append(torch.tensor([label], dtype=torch.long))

all_left = torch.cat(all_left, dim=0)
all_right = torch.cat(all_right, dim=0)
all_labels = torch.cat(all_labels, dim=0)

labeled_A = CoTrainingDataset(all_left, all_labels)
labeled_B = CoTrainingDataset(all_right, all_labels)

unlabeled_loader = DataLoader(unlabeled_training_set, batch_size=64, shuffle=False, generator=generator)

# Initially train the models in preparation for co-training
loader_A = DataLoader(labeled_A, batch_size=64, shuffle=True, generator=generator)
loader_B = DataLoader(labeled_B, batch_size=64, shuffle=True, generator=generator)

initial_metrics_A = train(model_A, device, 5, loader_A, loader_A, optimizer_A, criterion, int(args.overfitting_window), float(args.lambda_l2))
initial_metrics_B = train(model_B, device, 5, loader_B, loader_B, optimizer_B, criterion, int(args.overfitting_window), float(args.lambda_l2))

# Co-training parameters
co_training_iterations = 7
high_confidence_threshold = 0.9
low_confidence_threshold = 0.6

co_train(model_A, model_B, device, generator, labeled_A, labeled_B, unlabeled_loader, optimizer_A, optimizer_B, criterion, int(args.overfitting_window), int(args.lambda_l2), co_training_iterations, high_confidence_threshold, low_confidence_threshold)
accuracy = predict_with_both_models(model_A, model_B, test_loader, device)
print(f"Co-Training Final Accuracy: {accuracy}")
# TODO: Try to find better ways to improve overfitting

# TODO: Consistency Regularization (if I have time)