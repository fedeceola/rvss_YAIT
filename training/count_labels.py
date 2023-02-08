import torch
from torch.utils.data import DataLoader, random_split
from training.steerDS import SteerDataSet

transform = transforms.Compose([
        transforms.ToTensor(),

    ])
ds = SteerDataSet(
    os.path.join(os.getcwd(), "data"),
    ".jpg",
    transform
)

print("The dataset contains %d images " % len(ds))

trainset, valset = random_split(ds, [args.train_split, args.val_split])

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
dict_classes = {'0': 0, '-0.1': 0, '-0.2': 0,
'-0.3': 0,
'-0.4': 0,
'-0.5': 0,
'0.1': 0,
'0.2': 0,
'0.3': 0,
'0.4': 0,
'0.5': 0}

for data in trainloader:
    label = data['steering']

    dict_classes[str(label)] += 1

print(dict_classes)
