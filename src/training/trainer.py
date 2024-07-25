
from typing import Dict
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger

from data.preprocessor import PreProcessor
from data.dataset import MovieSentimentDataset
from model.model import  SequenceClassifier

class Trainer(object):

    CONFIG_PATH = '/app/configs/config.yaml'

    def __init__(self):
        self.config = self._load_config()


    def run(self) -> None:
        L.seed_everything(self.config['training']['seed'])
        self._set_datasets()
        self._set_dataloaders()
        self._set_model()
        # self._set_callbacks()
        self._set_trainer()
        self._start_training()

    def _load_config(self) -> Dict:
        stream = open(self.CONFIG_PATH, 'r')
        docs = yaml.load_all(stream,Loader=yaml.SafeLoader)
        param_dict = dict()
        for doc in docs:
            for k, v in doc.items():
                param_dict[k] = v
        return param_dict

    def _start_training(self) -> None:
        if self.config['training']['auto_lr']:
            lr_finder = self.trainer.tuner.lr_find(
                self.model, self.train_loader, self.valid_loader)
            new_lr = lr_finder.suggestion()
            self.model.learning_rate = new_lr
            self.print('Found a starting LR of {}'.format(new_lr))
        self.trainer.fit(self.lightning_model, self.train_dataloader, self.valid_dataloader)

    def _set_trainer(self) -> None:
        self.trainer = L.Trainer(
            devices=self.config['training']['n_gpu'],

            max_epochs=self.config['training']['epochs'],
            logger= self._get_logger()

        )

    def _set_model(self) -> None:
      
        self.lightning_model = LightningSequenceClassifier(self.config)

    def _set_datasets(self) -> None:
        preprocessor = PreProcessor()
        self.train_dataset = MovieSentimentDataset('train', preprocessor)
        self.valid_dataset = MovieSentimentDataset('valid', preprocessor)

    def _set_dataloaders(self) -> None:
        self.train_dataloader = DataLoader(self.train_dataset, collate_fn=self.train_dataset.preprocessor.collate,
                                       batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=self.config['training']['num_workers'])
        self.valid_dataloader = DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.preprocessor.collate,
                                       batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=self.config['training']['num_workers'])

    def _get_logger(self) -> MLFlowLogger:
        return MLFlowLogger(experiment_name=self.config['mlflow']['experiment_name'],tracking_uri=self.config['mlflow']['uri'],run_name=self.config['mlflow']['run_name'])

    def _set_callbacks(self) -> None:
        self.callbacks = []

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['training']['local_trained_model_path'],
            filename='best',
            save_top_k=1,
            verbose=True,
            save_last=True,
            monitor='val_loss',
            mode='min',
            every_n_epochs=self.config['training']['model_save_period']
        )
        self.callbacks.append(checkpoint_callback)
        if self.config['training']['lr_schedule']['scheduler']:
            lr_logging_callback = LearningRateMonitor(logging_interval='step')
            self.callbacks.append(lr_logging_callback)

class LightningSequenceClassifier(L.LightningModule):
    """
    Seq2seq Model Object
    """

    def __init__(self, config):
        """
        """
        super().__init__()
        self.config = config
        self.model = SequenceClassifier(config['model'])
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
        Optimizer
        Adam and Learning Rate Decay used.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['training']['learning_rate'])
        return optimizer
    
    def training_step(self, batch: Dict[str,torch.Tensor], batch_idx: int) -> torch.Tensor:
        '''
        Pytorch Lightning Trainer (training)
        '''
        x = batch['text']
        y = batch['label'].squeeze()
        output = self.model(x)
        loss = self.loss(output, y)
       
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

