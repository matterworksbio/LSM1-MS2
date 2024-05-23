import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models.lsm_recon import LSM
from msdatasets import MSDataset
from einops import rearrange


torch.set_float32_matmul_precision('medium')



config = {
    'dataset_path': '../datasets/train/final_merged_train.zarr',
    'batch_size': 448,
    'd_model': 1024,
    'heads': 16,
    'depth': 16,
    'lr': 1e-6,
    'checkpoint_path': None,
}


'''
Datamodule for MS data (pretraining)
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
        # Load main training set
        all_data = MSDataset(config['dataset_path'], mode='pretrain')
        # Load molecule/spectrum disjoint validation set
        self.valid_set = MSDataset('../datasets/val/val.zarr', mode='pretrain')
        train_set_size = int(len(all_data) * 0.95)
        valid_set_size = len(all_data) - train_set_size
        # Split main training set into training and validation sets
        self.train_set, self.disjoint_valid_set = data.random_split(all_data, [train_set_size, valid_set_size])
        # Load small CASMI validation set
        self.val_casmi = MSDataset('../datasets/test/casmi2017.zarr', mode='pretrain')
    
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
        loader1= DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )
        loader2= DataLoader(
            self.disjoint_valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )
        loader3= DataLoader(
            self.val_casmi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return [loader1, loader2, loader3]

'''
class SSModel: pytorch lightning model for pretraining
- inputs:
    - d_model: model dimension
    - heads: number of heads
    - depth: number of layers
    - lr: learning rate
    
- outputs:
    - model outputs:
        - mz1_logits: logits for the integer part of mz in each peak
        - mz2_logits: logits for the decimal part of mz in each peak
        - inty_logits: logits for the intensity in each peak
    
    - loss functions:
        - mz1_loss * alpha: cross entropy loss for mz1_logits * weight constant alpha
        - mz2_loss * beta: cross entropy loss for mz2_logits * weight constant beta
        - inty_loss * omega: cross entropy loss for inty_logits * weight constant omega
    
    - wandb logs:
        * {stage} E {train, val1, ..., val(n)}
        - train_loss: total loss for training set
        - val_loss: total loss for validation set(s)
        - {stage}_mz1_loss: mz1 loss for {stage} set
        - {stage}_mz2_loss: mz2 loss for {stage} set
        - {stage}_inty_loss: inty loss for {stage} set
        - {stage}_acc_mz1: mz1 accuracy for {stage} set
        - {stage}_acc_mz2: mz2 accuracy for {stage} set
        - {stage}_acc_inty: inty accuracy for {stage} set
        - {stage}_acc_masked_mz1: mz1 accuracy for {stage} set (masked tokens only)
        - {stage}_acc_masked_mz2: mz2 accuracy for {stage} set (masked tokens only)
        - {stage}_acc_masked_inty: inty accuracy for {stage} set (masked tokens only)
        - {stage}_top3_acc_masked_mz1: mz1 top 3 accuracy for {stage} set (masked tokens only)
        - {stage}_top3_acc_masked_mz2: mz2 top 3 accuracy for {stage} set (masked tokens only)
        - {stage}_top3_acc_masked_inty: inty top 3 accuracy for {stage} set (masked tokens only)
        
        
'''
class SSModel(pl.LightningModule):
    def __init__( self, d_model, heads, depth, lr, ms2prop=False):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.heads = heads
        self.depth = depth
        self.lr = lr
        self.model = LSM(d_model=self.d_model, heads=self.heads, depth=self.depth, mask_pct=0.25, ms2prop=ms2prop)

        # Define constants for loss functions
        self.alpha = 100.
        self.beta = 1.
        self.omega = 1.

        # Define loss functions
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
    
    '''
    forward pass: return logits
    '''
    def forward(self, mode, mz, inty):
        mz1_logits, mz2_logits, inty_logits, mask = self.model(mode, mz, inty)
        return mz1_logits, mz2_logits, inty_logits, mask
    
    '''
    Evaluation loop at end of each step,
        - stage: stage of training (train, val1, ..., val(n))
    - outputs: loss
    
    - notes:
        - calculates loss and logs metrics
        - crucially, this only calculate loss on non-zero (non-padded) peaks 
    '''
    def evaluate(self, batch, stage):
        precursor = batch['precursormz']
        mz = batch['mz']
        inty = batch['inty']
        
        # get logits
        mz1_logits, mz2_logits, inty_logits, mask = self(precursor, mz, inty) 
                        
        # ground truth intensity
        inty = inty.long()
        
        # get integer and decimal parts of mz for ground truth
        mz = torch.round((mz / 1_000) * (1e6 - 1)).long()
        gt_mz1 = mz//1000
        gt_mz2 = mz%1000
        
        # reshape for loss calculation
        gt_mz1 = rearrange(gt_mz1, 'b n -> (b n)')
        gt_mz2 = rearrange(gt_mz2, 'b n -> (b n)')
        inty = rearrange(inty, 'b n -> (b n)')
        
        mz1_logits = rearrange(mz1_logits, 'b n c -> (b n) c')
        mz2_logits = rearrange(mz2_logits, 'b n c -> (b n) c')
        inty_logits = rearrange(inty_logits, 'b n c -> (b n) c')

        mask = rearrange(mask, 'b n -> (b n)')        

        # get mask for non-zero peaks, then filter out zero peaks
        zeros_mask = inty > 0 

        gt_mz1 = gt_mz1[zeros_mask]
        gt_mz2 = gt_mz2[zeros_mask]
        inty = inty[zeros_mask]
        mz1_logits = mz1_logits[zeros_mask]
        mz2_logits = mz2_logits[zeros_mask]
        inty_logits = inty_logits[zeros_mask]
        mask = mask[zeros_mask]
        
        #calculate loss
        mz1_loss = self.criterion(mz1_logits, gt_mz1)
        mz2_loss = self.criterion(mz2_logits, gt_mz2)
        inty_loss = self.criterion(inty_logits, inty)
                
        self.log(f"{stage}_mz1_loss", mz1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_mz2_loss", mz2_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_inty_loss", inty_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        #Calculate MAE
        pred_mz1 = mz1_logits.argmax(dim=-1)
        pred_mz2 = mz2_logits.argmax(dim=-1)
        pred_inty = inty_logits.argmax(dim=-1)
        
        MAE_mz1 = torch.abs(pred_mz1 - gt_mz1).float().mean()
        MAE_mz2 = torch.abs(pred_mz2 - gt_mz2).float().mean()
        MAE_inty = torch.abs(pred_inty - inty).float().mean()
        
        #calculate accuracy
        acc_mz1 = (pred_mz1 == gt_mz1).float().mean()
        acc_mz2 = (pred_mz2 == gt_mz2).float().mean()
        acc_inty = (pred_inty == inty).float().mean()
        
        
        #masked tokens accuracy
        acc_masked_mz1 = (pred_mz1[mask] == gt_mz1[mask]).float().mean()
        acc_masked_mz2 = (pred_mz2[mask] == gt_mz2[mask]).float().mean()
        acc_masked_inty = (pred_inty[mask] == inty[mask]).float().mean()
        
        #masked tokens top 3 accuracy
        top3_mz1 = torch.topk(mz1_logits[mask], k=3, dim=-1).indices
        top3_mz2 = torch.topk(mz2_logits[mask], k=3, dim=-1).indices
        top3_inty = torch.topk(inty_logits[mask], k=3, dim=-1).indices
        

        # Check if the targets are in the top 3 preds
        correct_mz1 = top3_mz1.eq(gt_mz1[mask].view(-1, 1).expand_as(top3_mz1))
        top3_accuracy_mz1 = correct_mz1.any(dim=1).float().mean()
        
        correct_mz2 = top3_mz2.eq(gt_mz2[mask].view(-1, 1).expand_as(top3_mz2))
        top3_accuracy_mz2 = correct_mz2.any(dim=1).float().mean()
        
        correct_inty = top3_inty.eq(inty[mask].view(-1, 1).expand_as(top3_inty))
        top3_accuracy_inty = correct_inty.any(dim=1).float().mean()

        # Log metrics
        self.log(f"{stage}_acc_mz1", acc_mz1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc_mz2", acc_mz2, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc_inty", acc_inty, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log(f"{stage}_acc_masked_mz1", acc_masked_mz1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc_masked_mz2", acc_masked_mz2, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_acc_masked_inty", acc_masked_inty, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log(f"{stage}_top3_acc_masked_mz1", top3_accuracy_mz1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_top3_acc_masked_mz2", top3_accuracy_mz2, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_top3_acc_masked_inty", top3_accuracy_inty, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        
        self.log(f"{stage}_MAE_mz1", MAE_mz1, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_MAE_mz2", MAE_mz2, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_MAE_inty", MAE_inty, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        
        loss = self.alpha * mz1_loss + self.beta * mz2_loss + self.omega * inty_loss
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        
    
    def training_step(self, batch, batch_idx,):
        loss = self.evaluate(batch, stage='train')
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.evaluate(batch, stage='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
    

if __name__ == "__main__":
    print("initializing wandb")
    pl.seed_everything(42)
    # Initialize wandb
    wandb_logger = WandbLogger(
        project="",
        entity="",
        # id='2ju60nmw',
        config=config,
        log_model=True,
        mode="online",
    )
    print('---')
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    print(f"loading dataset from {config['dataset_path']}")
    # Load dataset
    dataset = MSDataModule(
        config["dataset_path"],
        config["batch_size"],
    )
    print("initializing model")
    # Initialize model
    model = SSModel(
        d_model=config["d_model"],
        heads=config["heads"],
        depth=config["depth"],
        lr=config["lr"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", dirpath='trained_models', filename=f"{model_name}_best", save_top_k=-1, mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0,1],
        precision='16-mixed',
        max_epochs=50,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.0,
        logger=wandb_logger,
        # accumulate_grad_batches=2
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)


    # Finish wandb run
    wandb_logger.experiment.finish()