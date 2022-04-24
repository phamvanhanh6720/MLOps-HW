from abc import ABC
from typing import Optional

from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.core import LightningDataModule


resized_size = 224
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((resized_size, resized_size)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])


class DatasetModule(LightningDataModule, ABC):
    def __init__(self, data_root='/tmp/cifar100',batch_size=32):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = CIFAR100(root=self.hparams.data_root, train=True, transform=transform, download=True)
        self.test_dataset = CIFAR100(root=self.hparams.data_root, train=False, transform=transform, download=False)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True
        )


if __name__ == '__main__':
    dm = DatasetModule()
    dm.setup()

    print(dm.test_dataset.classes)