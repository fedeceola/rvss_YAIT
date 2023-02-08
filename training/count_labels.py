import torch
from torch.utils.data import DataLoader, random_split
from steerDS import SteerDataSet
from torchvision import transforms


transform = transforms.Compose([
        transforms.ToTensor(),

    ])
ds = SteerDataSet(
    "/home/user/images/",
    0.35,
    ".jpg",
    transform
)

print("The dataset contains %d images " % len(ds))

trainset, valset = random_split(
        ds, [0.7, 0.3],
        generator=torch.Generator().manual_seed(10)
    )


trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
dict_classes = {'0.0': 0, '-0.1': 0, '-0.2': 0,
'-0.3': 0,
'-0.4': 0,
'-0.5': 0,
'0.1': 0,
'0.2': 0,
'0.3': 0,
'0.4': 0,
                '0.5': 0,
                '-0.0': 0}

for data in trainloader:
    label = data['steering']

    for i in range(label.shape[0]):
        dict_classes[str(round(label[i].item(),2))] += 1

print(dict_classes)
