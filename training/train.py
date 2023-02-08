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
from training.net import load_model
from training.steerDS import SteerDataSet


def main(args):
    # Load data
    if args.model_name == "Net":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            )
        ])
    elif args.model_name == "mobilenet_v2":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise NotImplementedError

    ds = SteerDataSet(
        os.path.join(os.getcwd(), "data"),
        args.crop_ratio,
        ".jpg",
        transform
    )
    print("The dataset contains %d images " % len(ds))

    trainset, valset = random_split(
        ds, [args.train_split, args.val_split], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create network
    net = load_model(args.model_name, args.feat_vect_dim)
    net.to(device)

    # Define loss and optim
    criterion = nn.MSELoss()

    if args.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-04)
    else:
        raise NotImplementedError

    phases = ["train", "val"]
    dataloader = {}
    dataloader["train"] = trainloader
    dataloader["val"] = valloader

    writer = SummaryWriter() 
    print("Logging stats at " + writer.log_dir)

    # Train NN
    for epoch in range(1, args.num_epochs+1, 1):  # loop over the dataset multiple times

        epoch_best_model = None
        best_val_loss_clipped = sys.float_info.max
        loss_epoch = {p: 0 for p in phases}
        loss_clipped_epoch = {p: 0 for p in phases}
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

                    loss = loss.item()

                    loss_epoch[p] += loss * b_size

                    with torch.set_grad_enabled(False):
                        # Log also clipped loss
                        outputs_clipped = torch.clip(
                            outputs, min=float(args.min_bound), max=float(args.max_bound)
                        )
                        loss_clipped = criterion(outputs_clipped, labels)
                        loss_clipped = loss_clipped.item()
                        loss_clipped_epoch[p] += loss_clipped * b_size

            writer.add_scalars(
                "Epoch_loss/train_val",
                {p: loss_epoch[p] / len(dataloader[p].dataset)},
                epoch-1
            )
            writer.add_scalars(
                "Epoch_clipped_loss/train_val",
                {p: loss_clipped_epoch[p] / len(dataloader[p].dataset)},
                epoch-1
            )

        if loss_clipped_epoch["val"] < best_val_loss_clipped:
            best_val_loss_clipped = loss_clipped
            epoch_best_model = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "loss_clipped": best_val_loss_clipped,
                    "model_state_dict": net.state_dict(),
                },
                os.path.join(writer.log_dir, "best_model.pth")
            )

    # TODO create test.py testing pipeline
    # TODO create script to compute mean and std training set
    # TODO create script visualize data from dataloader
    # TODO put avg pooling in the model
    # TODO put augmentation e.g. brightness, blur etc
    # TODO decrease learning rate if val loss does not decrease
    # TODO early stopping

    print('Finished Training')
    print("Best validation loss is {:.4f} obtained at epoch {}"
          .format(best_val_loss_clipped, epoch_best_model))


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "rvss_YAIT"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--feat_vect_dim", type=int, default=48048)
    parser.add_argument("--min_max_boundaries", type=str, default="-0.5,0.5")
    parser.add_argument("--model_name", type=str, default="Net")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--crop_ratio", type=float, default=0.3)
    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split

    assert len(args.min_max_boundaries.split(",")) == 2
    args.min_bound = args.min_max_boundaries.split(",")[0]
    args.max_bound = args.min_max_boundaries.split(",")[1]
    assert args.min_bound < args.max_bound

    assert args.crop_ratio >= 0 and args.crop_ratio < 1

    main(args)