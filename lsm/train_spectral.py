import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from msdatasets import MSDataset
# from train import SSModel
from pretrain_MAE import SSModel
from models.utils import jaccard_index 

torch.set_float32_matmul_precision('medium')

config = {
    'dataset_path': '../datasets/train/final_train.zarr',
    'batch_size': 224,
    'd_model': 1024,
    'heads': 16,
    'depth': 16,
    'mask_pct': 0.0,
    'out_emb_size': 512,
    'lr': 1e-6,
    'checkpoint_path': None# './trained_models/twilight-durian-195_best.ckpt', 
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
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        self.train_set = MSDataset(self.dataset_path, mode='spectral', tanimoto_path = '../datasets/tanimoto/train_tanimoto.pkl')
        self.valid_set = MSDataset('../datasets/val/val.zarr', mode='spectral', tanimoto_path = '../datasets/tanimoto/val_tanimoto.pkl')
        self.casmi = MSDataset('../datasets/test/casmi2017.zarr', mode='spectral', tanimoto_path = '../datasets/tanimoto/casmi2017_tanimoto.pkl')
        self.sampled_valid_set = MSDataset('../datasets/val/disjoint_val.zarr', mode='spectral',tanimoto_path = '../datasets/tanimoto/disjoint_val_tanimoto.pkl')

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=0,
        )
    
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
    
    - outputs:
        - logits: predicted properties
        - z: embeddings
    
    - loss: MSE loss of cosine similarity and tanimoto score
    
    - wandb metrics:
        * {stage} E {train, val1, ..., valn}
        - {stage}_loss: loss
        - {stage}_r2: r2 score
'''

class Finetune_SSSpectral(pl.LightningModule):
    def __init__( self, d_model, heads, depth, lr, mask_pct=0.0, out_emb_size=128, SSModel_path="path_to_pretrained_lsm.ckpt"):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.heads = heads
        self.depth = depth
        self.lr = lr
        self.mask_pct = mask_pct
        self.out_emb_size = out_emb_size
        
        if SSModel_path is not None:
            SSModel_path = "path_to_pretrained_lsm.ckpt"
            self.model = SSModel.load_from_checkpoint(SSModel_path)
        else:
            self.model = SSModel(d_model=d_model, heads=heads, depth=depth, lr=lr)
            
        #set model to encoder only
        self.model = self.model.model.encoder
        self.model.mask_pct = self.mask_pct
        
        self.embedding_head = nn.Sequential(
                # nn.Dropout(p=0.1),
                nn.Linear(d_model, out_emb_size),
                nn.ReLU(),
            )

        # Define loss functions
        self.criterion = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-06)


    '''
    forward pass: forward pass of model, returns embeddings of shape (batch_size, out_emb_size)
    '''
    def forward(self, precursormz, mz, inty):
        z = self.model(precursormz, mz, inty)
        #take meam of tokens  for embedding
        z = z[0].mean(dim=1)
        z = self.embedding_head(z)
        return z
    
    '''
    training step: calculates loss and r2 score
        - takes in spectrum of interest and randomly selected comparison spectrum
        - calculates cosine similarity and tanimoto score
        - calculates r2 score between cosine similarity and tanimoto score
        - calculates mse loss between cosine similarity and tanimoto score
    '''
    def training_step(self, batch, batch_idx):
        precursormz, mz, inty, fingerprint = batch['precursormz'], batch['mz'], batch['inty'], batch['fingerprint']
        precursormz2, mz2, inty2, fingerprint2 = batch['precursormz2'], batch['mz2'], batch['inty2'], batch['fingerprint2']

        # calculate embeddings for contrastive task
        z = self(precursormz, mz, inty)
        z_hat = self(precursormz2, mz2, inty2)
        
        cos = self.cos_sim(z, z_hat)
        tanimoto = jaccard_index(fingerprint, fingerprint2).to(cos.device)
    
        #calculate r2 bewteen cos and tanimoto scores
        epsilon = 1e-8

        numerator = torch.sum((tanimoto - cos) ** 2, axis=0)
        denominator = torch.sum((tanimoto - torch.mean(tanimoto, axis=0)) ** 2, axis=0) + epsilon
        r2 = (1 - (numerator / denominator)).mean()
        
        #mse
        mae = torch.mean(torch.abs(tanimoto-cos))
        
        #calculate loss (MAE)
        loss = self.criterion(cos, tanimoto)
        
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_r2", r2, sync_dist=True, prog_bar=True)
        self.log("train_mae", mae, sync_dist=True, prog_bar=True)
        return loss
    
    '''
    validation step: calculates loss and r2 score
        - takes in spectrum of interest and randomly selected comparison spectrum
        - calculates cosine similarity and tanimoto score
        - calculates r2 score between cosine similarity and tanimoto score
        - calculates mse loss between cosine similarity and tanimoto score
    '''
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        precursormz, mz, inty, fingerprint = batch['precursormz'], batch['mz'], batch['inty'], batch['fingerprint']
        precursormz2, mz2, inty2, fingerprint2 = batch['precursormz2'], batch['mz2'], batch['inty2'], batch['fingerprint2']
        
        # calculate embeddings for contrastive task
        z = self(precursormz, mz, inty)
        z_hat = self(precursormz2, mz2, inty2)

        cos = self.cos_sim(z, z_hat)
        tanimoto = jaccard_index(fingerprint, fingerprint2).to(cos.device)
    
        #calculate r2 bewteen cos and tanimoto scores
        epsilon = 1e-8

        numerator = torch.sum((tanimoto - cos) ** 2, axis=0)
        denominator = torch.sum((tanimoto - torch.mean(tanimoto, axis=0)) ** 2, axis=0) + epsilon
        r2 = (1 - (numerator / denominator)).mean()
        
        #calculate mse
        mae = torch.mean(torch.abs(tanimoto-cos))
        
        #calculate loss (MAE)
        loss = self.criterion(cos, tanimoto)
        
        self.log(f"val_loss", loss, sync_dist=True, prog_bar=True)
        self.log(f"val_r2", r2, sync_dist=True, prog_bar=True)
        self.log(f'val_mae', mae, sync_dist=True, prog_bar=True)
        return loss


    '''
    configure optimizers: sets up AdamW optimizer with OneCycleLR scheduler
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=25,
        )

        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
               },
            }




if __name__ == "__main__":
    pl.seed_everything(42)

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="",
        entity="",
        # id='2ju60nmw',
        config=config,
        log_model=True,
        mode="offline",
    )
    config = wandb_logger.experiment.config
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    # Load dataset
    dataset = MSDataModule(
        config["dataset_path"],
        config["batch_size"],
    )

    # Initialize model
    model = Finetune_SSSpectral(
        d_model=config["d_model"],
        heads=config["heads"],
        depth=config["depth"],
        lr=config["lr"],
        mask_pct=config['mask_pct'],
        out_emb_size=config['out_emb_size']
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_loss/dataloader_idx_0", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        strategy="ddp",
        precision='16-mixed',
        sync_batchnorm=True,
        use_distributed_sampler=True,
        max_epochs=20,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.,
        logger=wandb_logger,
        accumulate_grad_batches=4,
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)


    # Finish wandb run
    wandb_logger.experiment.finish()