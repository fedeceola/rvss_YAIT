from inspect import istraceback
import os
import time
import argparse
import sys

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from training.net import Net
from training.steerDS import SteerDataSet


def main(args):
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ds = SteerDataSet(
        os.path.join(os.getcwd(), "data"),
        ".jpg",
        transform
    )
    print("The dataset contains %d images " % len(ds))

    trainset, valset = random_split(ds, [args.train_split, args.val_split])
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create network
    net = Net(args.feat_vect_dim)
    net.to(device)

    # Define loss and optim
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    phases = ["train", "val"]
    dataloader = {}
    dataloader["train"] = trainloader
    dataloader["val"] = valloader

    writer = SummaryWriter() 
    print("Logging stats at " + writer.log_dir)

    # Train NN
    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        loss_epoch = {p: 0 for p in phases}
        for p in phases:

            is_training = True if p == "train" else False

            net.train(is_training)
            with torch.set_grad_enabled(is_training):

                for i, data in enumerate(dataloader[p], start=1):
                    inputs, labels = data['image'].to(device), data['steering'].to(device)
                    b_size = inputs.shape[0]

                    if is_training:
                        # zero the parameter gradients
                        optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs).squeeze()
                    if not len(outputs.shape):
                        outputs = outputs.unsqueeze(dim=0)
                    loss = criterion(outputs, labels)

                    if is_training:
                        loss.backward()
                        optimizer.step()

                    loss_epoch[p] += loss.item() * b_size

            writer.add_scalars(
                "Epoch_loss/train_val",
                {p: loss_epoch[p] / len(dataloader[p].dataset)},
                epoch
            )

    # TODO save best validation loss model
    # TODO add sigmoid to scale output in -0.5 +0.5

    print('Finished Training')


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "rvss_YAIT"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--feat_vect_dim", type=int, default=70224)
    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split

    main(args)