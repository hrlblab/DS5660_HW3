from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
from functions import *
from train import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define path
    traindir = f"data/cifar-10-mini/train"
    validdir = f"data/cifar-10-mini/val"
    #traindir = f"data/cifar-10-batches-py/train"
    #validdir = f"data/cifar-10-batches-py/val"
    # Change to fit hardware
    num_workers = 0

    batch_size = 8
    n_epochs = 10

    learning_rate = 0.001

    max_epochs_stop = 3
    print_every = 1
    # define image transformations
    # define transforms
    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # You can use transforms.RandomResizedCrop() for crop
                # You can use transforms.RandomRotation() for rotation
                # You can use transforms.ColorJitter() for colorjitter
                # You can use transforms.RandomHorizontalFlip() for flip

                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),

                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }
    # Datasets from folders

    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
            datasets.ImageFolder(root=validdir, transform=image_transforms['valid'])
    }

    # Dataloader iterators, make sure to shuffle

    dataloaders = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'valid': torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    validationiter = iter(dataloaders['valid'])

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)

    n_classes = len(categories)
    print('Length train: {} Length test: {}'.format(len(data['train']), len(data['valid'])))

    """


    train_data = datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    valid_data = datasets.ImageFolder(root=traindir, transform=image_transforms['valid'])

    print('Length train: {} Length test: {}'.format(len(train_data), len(valid_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(valid_loader)))

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)

    n_classes = len(categories)
    """

    # get some random training images
    dataiter = iter(dataloaders['train'])
    images, labels = iter(dataiter).__next__()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % categories[labels[j]] for j in range(batch_size)))

    # Define the network with pretrained models
    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #cover class to index
    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    # Set up your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    save_file_name = 'resnet18_model_best_model.pt'

    model = train(model,
                  criterion,
                  optimizer,
                  dataloaders['train'],
                  dataloaders['valid'],
                  save_file_name,
                  max_epochs_stop,
                  n_epochs,
                  print_every)

    #"""
    dataiter = iter(dataloaders['valid'])
    images, labels = iter(dataiter).__next__()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % categories[labels[j]] for j in range(batch_size)))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % categories[predicted[j]] for j in range(batch_size)))
    
    #"""