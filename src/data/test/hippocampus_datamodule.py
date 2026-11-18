from lightning import LightningDataModule

class SegDataModule(LightningDataModule):
    def __init__(self, data_dir="data/", batch_size=2, num_workers=0, pin_memory=False):
        super().__init__()
        self.save_hyperparameters(logger=False)