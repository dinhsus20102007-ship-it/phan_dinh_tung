from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.segmentation import DiceScore


class SegLitModule(LightningModule):
 
    def __init__(self, net, optimizer, scheduler, num_classes=2, compile= False):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"]) # net not recored as hyperparameter

        self.net = net

        # loss function
        self.ce = torch.nn.CrossEntropyLoss()


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_dice = DiceScore(num_classes=num_classes, include_background=False, input_format="index") #for val/test
        self.test_dice = DiceScore(num_classes=num_classes, include_background=False, input_format="index")
        # compare similarity between prediction & mask(real)
        self.val_dice_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def dice_loss(self, logits, y, eps= 1e-5): #for training(backprop)
        probs = torch.softmax(logits, dim=1)
        y1h = torch.nn.functional.one_hot(y, probs.shape[1]).movedim(-1, 1).float()
        dims = tuple(range(2, probs.ndim))
        inter = (probs * y1h).sum((2, 3))
        denom = probs.sum((2, 3)) + y1h.sum((2, 3))
        return 1 - (2 * inter + eps) / (denom + eps).mean()

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.ce(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y


    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, y = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_dice(preds, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_dice_best(self.val_dice.compute())
        self.log("val/dice_best", self.val_dice_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self.model_step(batch)
        self.test_loss(loss)
        self.test_dice(preds, y)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/dice", self.test_dice, on_step=False, on_epoch=True)

    def setup(self, stage):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss",
                                     "interval": "epoch", "frequency": 1}}


if __name__ == "__main__":
    _ = SegLitModule(None, None, None, None)
