import torch
import torch.nn as nn
from torchmetrics.regression import R2Score
import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightly.loss.vicreg_loss import VICRegLoss
from msdatasets import MSDataset
# from train import SSModel
from pretrain_MAE import SSModel

torch.set_float32_matmul_precision('medium')

PATH = '' # path to data

config = {
    'dataset_path': f'{PATH}/datasets/train/final_train.zarr',
    'batch_size':  512,
    'd_model': 1024,
    'heads': 16,
    'depth': 16,
    'lr': 0.00025,
    'mask_pct': 0.0,
    'checkpoint_path': None, #"trained_models/youthful-dragon-265_best.ckpt",
    'ms2prop': False, 
    'dataset_size': 1.,
    'supervised': False,
    'linear_probing': False,
}


'''
Datamodule for MS data (finetuning)
- inputs: 
    - dataset_path: path to the dataset
    - batch_size: batch size
    
- outputs:
    - train_dataloader: dataloader for training set
    - val_dataloader: dataloader(s) for validation set
'''

class MSDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size, dataset_size=1., linear_probing=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.linear_probing = linear_probing
        

    def prepare_data(self):
        pass

    '''
    Setup: 
        - Loads train, validation, and test sets
    '''
    def setup(self, stage=None):
        mode = 'property'
        if self.linear_probing:
            mode = "lienar probing"

        all_data = MSDataset(self.dataset_path, mode=mode, train_minmax_path = f'{PATH}/datasets/processed_data/norm_values.csv',)
        train_set_size = int(len(all_data) * self.dataset_size)
        valid_set_size = len(all_data) - train_set_size
        self.train_set, _ = data.random_split(all_data, [train_set_size, valid_set_size])
        print(f"Train set size: {len(self.train_set)}")
    
        self.valid_set = MSDataset(f'{PATH}/datasets/val/val.zarr', mode=mode, train_minmax_path = f'{PATH}/datasets/processed_data/norm_values.csv')
        self.casmi = MSDataset(f'{PATH}/datasets/test/casmi2017.zarr', mode=mode, train_minmax_path = f'{PATH}/datasets/processed_data/norm_values.csv')
        self.sampled_valid_set = MSDataset(f'{PATH}/datasets/val/disjoint_val.zarr', mode=mode, train_minmax_path = f'{PATH}/datasets/processed_data/norm_values.csv')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
        )
    
    #returns list of 3 different validation dataloaders that we are using
    def val_dataloader(self):
        loader1 =  DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=True, #set shuffle to true, since running into r^2 collaps otherwise
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )
        loader2 = DataLoader(
            self.casmi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        loader3 = DataLoader(
            self.sampled_valid_set,
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=0,
        )
            
        return [loader1, loader2, loader3]

'''
Finetune_SSModel Class:
    - Finetunes a pretrained model on a property prediction task
    
    - inputs:
        - d_model: dimension of model
        - heads: number of heads
        - depth: number of layers
        - lr: learning rate
        - mask_pct: percent of spectra to mask out
        - num_properties: number of properties to predict
        - SSModel_path: path to pretrained model
        - ms2prop: whether to use ms2prop hyperparameters (32 heads, 6 depth, 512 d_model)
        - supervised: whether to use supervised or unsupervised model (if supervised, initialize new base encoder)
        - linear_probing: whether to freeze encoder and only train classification head
    
    - outputs:
        - logits: predicted properties
        - z: embeddings
    
    - loss: MSE loss
    
    - wandb metrics:
        * {stage} E {train, val1, ..., valn}
        - {stage}_loss: loss
        - {stage}_acc: accuracy
        - {stage}_MAE: mean absolute error
        - {stage}_r2: r^2 score
'''
class Finetune_SSModel(pl.LightningModule):
    def __init__( self, d_model, heads, depth, lr, mask_pct=0.0, num_properties=209, SSModel_path='path_to_pretrained_lsm_checkpoint', ms2prop=False, supervised=False, linear_probing=False):
        super().__init__(),
        self.save_hyperparameters()
        self.d_model = d_model
        self.heads = heads
        self.depth = depth
        self.lr = lr
        self.num_properties = num_properties
        self.mask_pct = mask_pct
        self.ms2prop = ms2prop
        
        
        #load pretrained model or initialize new model
        if supervised:
            self.model = SSModel(d_model=d_model, heads=heads, depth=depth, lr=lr)
        elif SSModel_path is not None and ms2prop == False:
            SSModel_path = "path_to_pretrained_lsm_checkpoint"
            self.model = SSModel.load_from_checkpoint(SSModel_path)
        else:
            print(f'using MS2Prop: {self.ms2prop}')
            self.model = SSModel(d_model=d_model, heads=heads, depth=depth, lr=lr, ms2prop=True)
        
        #extract only the encoder
        self.model = self.model.model.encoder
        self.model.mask_pct = self.mask_pct 
        
        #if linear probing, freeze all of the encoder
        if linear_probing:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.d_model, self.num_properties),
        )

        # Define loss functions
        self.criterion = nn.MSELoss()

    '''
    Forward pass: returns logits and embeddings
    '''
    def forward(self, precursormz, mz, inty):
        z = self.model(precursormz, mz, inty)
        if self.ms2prop:
            z = z[0][:, 0, :]
        else:
            z = z[0].mean(dim=1)
        logits = self.classification_head(z)
        return logits, z
    
    '''
    Training step:
        - inputs: batch
        
        - outputs: loss, accuracy, MAE, r^2 score
    '''
    def training_step(self, batch, batch_idx):
        precursormz = batch['precursormz']
        mz = batch['mz']
        inty = batch['inty']
        gt_feats = batch['y_feats']
        y_feats, _ = self(precursormz, mz, inty)
        loss = self.criterion(gt_feats, y_feats)
        accuracy = (gt_feats == y_feats).sum() / gt_feats.shape[0]
        MAE = torch.mean(torch.abs(gt_feats - y_feats))
        # Calculating R^2
        epsilon = 1e-6

        numerator = torch.sum((gt_feats - y_feats) ** 2, axis=0)
        denominator = torch.sum((gt_feats - torch.mean(gt_feats, axis=0)) ** 2, axis=0) + epsilon
        r2 = (1 - (numerator / denominator)).mean()
        
        self.log("train_MAE", MAE, sync_dist=True, prog_bar=True)
        self.log("train_acc", accuracy, sync_dist=True, prog_bar=True)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_r2", r2, sync_dist=True, prog_bar=True)
        return loss
    
    '''
    Validation step:
        - inputs: batch
        
        - outputs: loss, accuracy, MAE, r^2 score
    '''
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        precursormz = batch['precursormz']
        mz = batch['mz']
        inty = batch['inty']
        gt_feats = batch['y_feats']
        y_feats, _ = self(precursormz, mz, inty)
        loss = self.criterion(gt_feats, y_feats)
        
        accuracy = (gt_feats == y_feats).sum() / gt_feats.shape[0]
        MAE = torch.mean(torch.abs(gt_feats - y_feats))
        # Calculating R^2
        epsilon = 1e-6

        numerator = torch.sum((gt_feats - y_feats) ** 2, axis=0)
        denominator = torch.sum((gt_feats - torch.mean(gt_feats, axis=0)) ** 2, axis=0) + epsilon
        r2 = (1 - (numerator / denominator)).mean()
        
        self.log(f"val_MAE", MAE, sync_dist=True, prog_bar=True)
        self.log(f"val_acc", accuracy, sync_dist=True, prog_bar=True)
        self.log(f"val_loss", loss, sync_dist=True, prog_bar=True)
        self.log(f"val_r2", r2, sync_dist=True, prog_bar=True)
        return loss

    '''
    Optimizer: AdamW with OneCycleLR scheduler
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return {"optimizer": optimizer,
            }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        precursormz = batch['precursormz']
        mz = batch['mz']
        inty = batch['inty']
        embedding = self.model.encoder(precursormz, mz, inty)
        return embedding



if __name__ == "__main__":
    pl.seed_everything(42)

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="",
        entity="",
        # id='2ju60nmw',
        config=config,
        log_model=False,
        mode="offline",
    )
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    # Load dataset
    dataset = MSDataModule(
        config["dataset_path"],
        config["batch_size"],
        config["dataset_size"],
    )

    # Initialize model
    model = Finetune_SSModel(
        d_model=config["d_model"],
        heads=config["heads"],
        depth=config["depth"],
        lr=config["lr"],
        mask_pct=config["mask_pct"],
        ms2prop=config["ms2prop"],
        supervised=config["supervised"],
        linear_probing=config["linear_probing"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_MAE/dataloader_idx_0", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        strategy="ddp_find_unused_parameters_true",
        precision='16-mixed',
        sync_batchnorm=True,
        use_distributed_sampler=True,
        max_epochs=50,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.,
        logger=wandb_logger,
        # accumulate_grad_batches=2,
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)


    # Finish wandb run
    wandb_logger.experiment.finish()