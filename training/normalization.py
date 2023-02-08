import torch
from torch.utils.data import DataLoader, random_split
from training.steerDS import SteerDataSet

def main(args):


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



    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in trainloader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
            cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
            cnt + nb_pixels)
        cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(
            snd_moment - fst_moment ** 2)



        print("mean and std: \n", mean, std)


if __name__ == "__main__":
    launch_dir = os.path.basename(os.getcwd())
    expected = "rvss_YAIT"
    assert launch_dir == expected

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_split", type=float, default=0.7)

    args = parser.parse_args()

    assert args.train_split > 0 and args.train_split < 1
    args.val_split = 1 - args.train_split


    main(args)
