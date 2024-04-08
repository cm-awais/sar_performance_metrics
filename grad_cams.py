# Imports

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import multiprocessing as mp
from torchvision import models
from copy import deepcopy
import warnings
import csv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings('ignore')


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



def load_data(address, batch_size=64, train=True):
  # Load Fusar dataset
  if train:
    dataset = ImageFolder(root=address, transform=transform_train)
  else: 
    dataset = ImageFolder(root=address, transform=transform_test)

  # Create a dictionary of class names
  class_names = {i: classname for i, classname in enumerate(dataset.classes)}

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2,  # Experiment with different values as recommended above
                            # pin_memory=False, # if torch.cuda.is_available() else False,
                            persistent_workers=True)
  print("Top classes indices:", class_names)

  return data_loader

class VGGModel(nn.Module):
  def __init__(self, pretrained=False):
    super(VGGModel, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features  # Use VGG16 features
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedVGG(nn.Module):
  def __init__(self, pretrained=True):
    super(FineTunedVGG, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features
    for param in self.features.parameters():
      param.requires_grad = False  # Freeze pre-trained layers

    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling

    # self.classifier = nn.Sequential(*list(self.features.classifier.children())[:-1])  # Use all but last layer
    # self.classifier.add_module('final', nn.Linear(self.classifier[-1].in_features, 3))  # Add new final layer
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class ResNetModel(nn.Module):
  def __init__(self, pretrained=False):
    super(ResNetModel, self).__init__()
    resnetf = models.resnet50(pretrained=pretrained)

    self.features = nn.Sequential(*list(resnetf.children())[:-1]) # Use ResNet50 features
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
    self.classifier = nn.Sequential(
      nn.Linear(10 * 10, 4096),  # Adjust based on input size
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 3)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedResNet(nn.Module):
  def __init__(self, pretrained=True):
    super(FineTunedResNet, self).__init__()

    # Load pre-trained ResNet50 model
    resnetf = models.resnet50(pretrained=pretrained)

    self.features = nn.Sequential(*list(resnetf.children())[:-1])

    # Freeze pre-trained layers
    for param in self.features.parameters():
      param.requires_grad = False

    # Replace final layer and adjust for grayscale input
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))  # Global Average Pooling
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # self.fc = nn.Linear(self.features.fc.in_features, 3)  # Replace final layer
    self.classifier = nn.Sequential(
            nn.Linear(10 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # 3 output classes
        )

  def forward(self, x):
    # Convert grayscale image to 3-channel tensor (assuming single channel)
    # print(x.size(1))
    if x.size(1) == 1:  # Check if input has 1 channel
      x = x.repeat(1, 3, 1, 1)  # Duplicate grayscale channel 3 times
    # print(x.size(1))

    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class ImprovedCNNModel(nn.Module):
  def __init__(self):
    super(ImprovedCNNModel, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),  # Add Batch Normalization for better stability
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.classifier = nn.Sequential(
      nn.Linear(4096 * 7 * 7, 4096),  # Adjust for final feature map size
      nn.ReLU(inplace=True),
      nn.Dropout(p=0.5),  # Adjust dropout probability
      nn.Linear(4096, 3)
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 3)  # 3 output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

fusar_path = "fusar_split"
open_sar_path = "opensarship_u1"
mix_path = "mix_u"
batch_size = 64
exp_images_s = 15

csv_res = []
csv1 = []

print("Loading Data")

# fusar_train_loader = load_data(fusar_path + "/train", batch_size=batch_size)
fusar_test_loader = load_data(fusar_path + "/test", batch_size=batch_size)

# open_sar_train_loader = load_data(open_sar_path + "/train", batch_size=batch_size)
open_sar_test_loader = load_data(open_sar_path + "/test", batch_size=batch_size)

# mix_train_loader = load_data(mix_path + "/train", batch_size=batch_size)
mix_test_loader = load_data(mix_path + "/test", batch_size=batch_size)

print("Data Loaded")

exp_dir = "Exp_images"
datasets = {"Fusar": fusar_test_loader,
            "OpenSARShip": open_sar_test_loader,
            "Mixed_fusar": mix_test_loader}

# datasets = {"OpenSARShip": open_sar_test_loader}

models = {"CNN": ImprovedCNNModel(),
          "VGG": VGGModel(),
          "Fine_VGG": VGGModel(),
          "ResNet": ResNetModel(),
          "Fine_Resnet": ResNetModel()}

trained_dir = "trained_models"

for dataset_name, dataset_loader in datasets.items():
  cl1 = []
  # ct1 = "Cargo"
  cl2 = []
  # ct2 = "Fishing"
  cl3 = []
  # ct3 = "Tanker"

  for batch_idx, data in enumerate(dataset_loader):
    data, target = data[0].to(device), data[1]
    targets = target.numpy()
    if len(cl1) == exp_images_s and len(cl2) == exp_images_s and len(cl3)==exp_images_s:
        break
    for id, targ in enumerate(targets):
      if targ == 0:
        if len(cl1)<exp_images_s:
          cl1.append(data[id])
      if targ == 1:
        if len(cl2)<exp_images_s:
          cl2.append(data[id])
      if targ == 2:
        if len(cl3)<exp_images_s:
          cl3.append(data[id])
      if len(cl1) == exp_images_s and len(cl2) == exp_images_s and len(cl3)==exp_images_s:
        break

  cl1.extend(cl2)
  cl1.extend(cl3)
  classes_tensor = torch.stack(cl1, dim=0)

  for model_name, model in models.items():
    model_filename = model_name+"_"+dataset_name + ".pt"
    trained_path = os.path.join(trained_dir, model_filename)
    expl_path = os.path.join(exp_dir, dataset_name, model_name)
    os.makedirs(expl_path, exist_ok=True)
    model.to(device)
    model.load_state_dict(torch.load(trained_path))
    model.eval()

    target_layer = [model.features[-1]]

    gradcam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cams = gradcam(classes_tensor)

    for cam_id, cam in enumerate(grayscale_cams):
      class_n = ""
      if cam_id<exp_images_s:
        class_n = "/Cargo"
      elif cam_id<exp_images_s*2:
        class_n = "/Fishing"
      else:
        class_n = "/Tanker"
      imag = cl1[cam_id].permute(1, 2, 0)
      if imag.ndim == 3 and imag.shape[0] == 1:
        imag = imag.view(imag.shape[1:])  # Remove redundant channel dimension

      # Convert tensor to NumPy array
      image_array = imag.cpu().numpy()

      vis = show_cam_on_image(image_array, cam, use_rgb=True)
      h_im = Image.fromarray(vis)

      # Saving heatmap Image
      f_name = expl_path + class_n+"_hm"+str(cam_id)+".png"
      h_im.save(f_name)

      if image_array.shape[0] == 1:
        image_array = image_array.squeeze(axis=0)

      image_array = (image_array * 255).astype(np.uint8)  # Ensure uint8

      # Saving original corrosponding Image
      o_im = Image.fromarray(image_array, mode='RGB')
      o_name = expl_path + class_n+"_o"+str(cam_id)+".png"
      o_im.save(o_name)