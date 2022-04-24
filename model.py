from abc import ABC
from typing import Any

import torch
from sklearn.metrics import accuracy_score
import torchvision.models as models
from torchvision import transforms
from pytorch_lightning import LightningModule


class Cifar100Model(LightningModule, ABC):
    def __init__(self, learning_rate=5e-4, dropout_rate=0.3, threshold=0.5, n_classes=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.efficientnet_b0(pretrained=True)
        self.head_cls = torch.nn.Sequential(
            torch.nn.Dropout(p=self.hparams.dropout_rate),
            torch.nn.Linear(1000, 256),
            torch.nn.Dropout(p=self.hparams.dropout_rate),
            torch.nn.Linear(256, self.hparams.n_classes)
        )

    def forward(self, x) -> Any:
        feature = self.model(x)
        logit = self.head_cls(feature)

        return logit

    def training_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        labels = labels.to(torch.long)

        logits = self.forward(imgs)

        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        preds = torch.argmax(logits, dim=-1)

        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        labels = labels.to(torch.long)

        logits = self.forward(imgs)
        print(logits.shape)

        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        preds = torch.argmax(logits, dim=-1)

        return {"loss": loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        imgs, labels = batch[0], batch[1]
        labels = labels.to(torch.long)

        logits = self.forward(imgs)

        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        preds = torch.argmax(logits, dim=-1)

        return {"loss": loss, "preds": preds, "labels": labels}

    def training_epoch_end(self, outputs):
        predictions = []
        labels = []
        for x in outputs:
            predictions.extend(x['preds'].detach().cpu().numpy().tolist())
            labels.extend(x['labels'].detach().cpu().numpy().tolist())

        acc = accuracy_score(labels, predictions)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        predictions = []
        labels = []
        for x in outputs:
            predictions.extend(x['preds'].detach().cpu().numpy().tolist())
            labels.extend(x['labels'].detach().cpu().numpy().tolist())

        acc = accuracy_score(labels, predictions)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    def test_epoch_end(self, outputs):
        predictions = []
        labels = []
        for x in outputs:
            predictions.extend(x['preds'].detach().cpu().numpy().tolist())
            labels.extend(x['labels'].detach().cpu().numpy().tolist())

        acc = accuracy_score(labels, predictions)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
