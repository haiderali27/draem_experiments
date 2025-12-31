import argparse
from types import SimpleNamespace
from lightning import Trainer
import torch
from dataloader.data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
import os
from models.draem.draem import DraemModel
import urllib.request
import tarfile

def get_device(cuda_device:int=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_device}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(args: SimpleNamespace):
    device = get_device()
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(f"{args.anomaly_path}/dtd"):
        archive_path = os.path.join(args.anomaly_path, "dtd-r1.0.1.tar.gz")
        urllib.request.urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=args.anomaly_path)
        os.remove(archive_path)

    dataset = MVTecDRAEMTrainDataset(
        os.path.join(args.dataset_path, args.object_name, "train/good/"),
        f'{args.anomaly_path}/dtd/images',
        resize_shape=[256, 256],
        ignore_black_region=args.ignore_black_region
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    model = DraemModel(config={"device": device,
                               "reconstructive_network_name": args.reconstruction_network_name,
                               "reconstruction_loss": args.reconstruction_loss,
                               "learning_rate": args.learning_rate,
                               "epochs": args.epochs,
                               "checkpoint_path": args.checkpoint_path
                               }).to(device)
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs
    )
    trainer.fit(model, dataloader)

    return trainer, model

def test(args: SimpleNamespace):
    if args.load_checkpoints:
        device = get_device()
        trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs
        )
        model = DraemModel(config={"device": device,
                               "load_checkpoints": True,
                               }).to(device)
    else:
        trainer, model = train(args=args)
    dataset = MVTecDRAEMTestDataset(os.path.join(args.dataset_path, args.object_name, "test/"), resize_shape=[256, 256])
    dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)
    trainer.test(model, dataloader)

def add_subparsers(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--object-name', '-o',
        type=str,
        default='cars',
        help='object name of dataset')
    train_parser.add_argument(
        '--reconstruction-network-name',
        type=str,
        choices=['Simple-Encoder', 'VAE-Encoder', 'Attention-Encoder', 'VAE-Attention-Encoder'],
        default='Simple-Encoder',
        help='reconstruction network name'
    )
    train_parser.add_argument(
        '--reconstruction-loss',
        type=str,
        choices=['SSIM', 'LPIPS', 'MMSIM'],
        default='SSIM',
        help='reconstruction network loss function'
    )
    train_parser.add_argument('--batch-size', '-s',
        type=int,
        default=8,
        help='data batch size')
    train_parser.add_argument('--learning-rate', '-l',
        type=float,
        default=0.0001,
        help='learning rate')
    train_parser.add_argument('--epochs', '-e',
        type=int,
        default=700,
        help='epochs')
    train_parser.add_argument('--dataset-path', '-p',
        type=str,
        default=f'{os.getcwd()}/dataset',
        help='path of dataset')
    train_parser.add_argument('--anomaly-path', '-a',
        type=str,
        default=f'{os.getcwd()}',
        help='path of images that are used for creating augmented anomalies')
    train_parser.add_argument('--ignore-black-region',
        type=bool,
        default=True,
        help='ignores augmentation on pixels with the value of 0')
    train_parser.add_argument('--checkpoint-path', '-c',
        type=str,
        default=f'{os.getcwd()}/checkpoints',
        help='training weights are saved in this path')
    train_parser.add_argument('--log-path',
        type=str,
        default=f'{os.getcwd()}/logs',
        help='path of logs')
    train_parser.set_defaults(func=train)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--object-name', '-o',
        type=str,
        default='cars',
        help='object name of dataset')
    test_parser.add_argument(
        '--reconstruction-network-name',
        type=str,
        choices=['Simple-Encoder', 'VAE-Encoder', 'Attention-Encoder', 'VAE-Attention-Encoder'],
        default='Simple-Encoder',
        help='reconstruction network name'
    )
    test_parser.add_argument(
        '--reconstruction-loss',
        type=str,
        choices=['SSIM', 'LPIPS', 'MMSIM'],
        default='SSIM',
        help='reconstruction network loss function'
    )
    test_parser.add_argument('--batch-size', '-s',
        type=int,
        default=8,
        help='data batch size')
    test_parser.add_argument('--learning-rate', '-l',
        type=float,
        default=0.0001,
        help='learning rate')
    test_parser.add_argument('--epochs', '-e',
        type=int,
        default=700,
        help='epochs')
    test_parser.add_argument('--dataset-path', '-p',
        type=str,
        default=f'{os.getcwd()}/dataset',
        help='path of dataset')
    test_parser.add_argument('--anomaly-path', '-a',
        type=str,
        default=f'{os.getcwd()}',
        help='path of images that are used for creating augmented anomalies')
    test_parser.add_argument('--ignore-black-region',
        type=bool,
        default=True,
        help='ignores augmentation on pixels with the value of 0')
    test_parser.add_argument('--checkpoint-path', '-c',
        type=str,
        default=f'{os.getcwd()}/checkpoints',
        help='training weights are saved in this path')
    test_parser.add_argument('--log-path',
        type=str,
        default=f'{os.getcwd()}/logs',
        help='path of logs')
    test_parser.add_argument('--load-checkpoints',
        type=bool,
        default=False,
        help='use pre-defined weights from checkpoints')
    test_parser.set_defaults(func=test)

    return parser

def main() -> None:
    parser = add_subparsers(argparse.ArgumentParser())
    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    kwargs = {key: value for key, value in args._get_kwargs() if key != 'func'}
    kwargs = SimpleNamespace(**kwargs)
    args.func(kwargs)

main()
