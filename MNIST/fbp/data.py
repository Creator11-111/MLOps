import os
from typing import Optional

import lightning.pytorch as pl
import torch
from dvc.fs import DVCFileSystem
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def check_files() -> bool:
    files_paths = [
        "data/MNIST/raw/t10k-images-idx3-ubyte",
        "data/MNIST/raw/t10k-labels-idx1-ubyte",
        "data/MNIST/raw/train-images-idx3-ubyte",
        "data/MNIST/raw/train-labels-idx1-ubyte",
    ]
    result = True
    for path in files_paths:
        result &= os.path.exists(path)

    return result


def load_data_from_dvc():
    fs = DVCFileSystem()
    fs.get("data", "data", recursive=True)


class MyDataModule(pl.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and
    transforms. The main advantage is consistent data splits, data preparation and
    transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP
    """

    def __init__(
        self,
        val_size: float,
        dataloader_num_wokers: int,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.val_size = val_size
        self.dataloader_num_wokers = dataloader_num_wokers
        self.batch_size = batch_size

    def prepare_data(self):
        """Use this to download and prepare data. Downloading and saving data with
        multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures this method is called only within a single process, so you can
        safely add your downloading logic within.

        Warning::
            DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            dm.prepare_data()
            initialize_distributed()
            dm.setup(stage)
            dm.train_dataloader()
            dm.val_dataloader()
            dm.test_dataloader()
            dm.predict_dataloader()
        """
        if not check_files():
            load_data_from_dvc()

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.

        setup is called from every process across all the nodes. Setting state here is
        recommended.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        mnist_train = MNIST("./data", train=True, download=False, transform=transform)
        mnist_train = [mnist_train[i] for i in range(2200)]

        self.train_dataset, self.val_dataset = random_split(mnist_train, [2000, 200])

        mnist_test = MNIST("./data", train=False, download=False, transform=transform)
        self.mnist_test = [mnist_test[i] for i in range(3000, 4000)]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """An iterable or collection of iterables specifying training samples.

        For more information about multiple dataloaders,
        see this: https://pytorch-lightning.readthedocs.io/en/1.0.8/multiple_loaders.html

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs`
        to a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary
            hardware. There is no need to set it yourself.
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_wokers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """An iterable or collection of iterables specifying validation samples.

        For more information about multiple dataloaders,
        see this: https://pytorch-lightning.readthedocs.io/en/1.0.8/multiple_loaders.html

        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs`
        to a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary
            hardware. There is no need to set it yourself.
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass
