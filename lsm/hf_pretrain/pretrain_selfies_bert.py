import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
# from train import SSModel
from transformers import (
    RobertaForMaskedLM,
    DataCollatorWithPadding,
    AutoTokenizer,
)
from transformers import RobertaConfig, RobertaForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import selfies as sf

torch.set_float32_matmul_precision('medium')

config = {
    'dataset_path': '/datasets/hf_smiles/zinc100m.parquet', #path to selfies parquet
    'batch_size': 256,
    'd_model': 1024,
    'lr': 1e-5,
    'mask_pct': 0.15,
    'checkpoint_path': None 
}


class SMILESDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
     ):
        self.df = pd.read_parquet(dataset_path)

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        return {'selfies': self.df.iloc[idx]['SELFIES'], 'smiles': self.df.iloc[idx]['smiles']}

class MSDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def prepare_data(self):
        # Implement if necessary, e.g., downloading, tokenizing
        pass

    def setup(self, stage=None):
        # Assuming SMILESDataset is properly defined elsewhere
        all_data = SMILESDataset(self.dataset_path)
        
        # Subset 5% of the train set for validation and test
        train_set_size = int(0.95 * len(all_data))
        val_test_set_size = len(all_data) - train_set_size

        # Ensure even split for validation and test sets
        val_set_size = (val_test_set_size + 1) // 2
        test_set_size = val_test_set_size - val_set_size

        # Perform the actual splitting
        self.train_set, val_test_set = data.random_split(all_data, [train_set_size, val_test_set_size])
        self.valid_set, self.test_set = data.random_split(val_test_set, [val_set_size, test_set_size])

        print(len(self.train_set), len(self.valid_set), len(self.test_set))

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
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False, 
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )
            


class MolBert(pl.LightningModule):
    def __init__( self, lr, mask_pct=0.15, **kwargs): 
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.mask_pct = mask_pct
        
        # Use the specified tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zjunlp/MolGen-large", max_len=256
        )
        self.collator = DataCollatorWithPadding(self.tokenizer, padding=True, return_tensors="pt")

        # Initialize Roberta configuration with hyperparameters matching roberta-large
        config = RobertaConfig(
            vocab_size=len(self.tokenizer.get_vocab()),
            max_position_embeddings=256,
            num_attention_heads=8,
            num_hidden_layers=8,
            type_vocab_size=1,
            hidden_size=1024,  
        )
        
        # Initialize the RoBERTa model from scratch
        self.model = RobertaForMaskedLM(config=config)

        #losses
        self.Cross_Entropy = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=output_hidden_states)
    
    def training_step(self, batch, batch_idx):
        selfies = batch['selfies']
        batch_size = len(selfies)

        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized_selfies['input_ids'].clone()
        mask_indices = torch.rand(tokenized_selfies['input_ids'].shape) < self.mask_pct
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.pad_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.eos_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.bos_token_id
        tokenized_selfies['input_ids'][mask_indices] = self.tokenizer.mask_token_id

        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        labels[~mask_indices] = -100
        # Apply mask where mask_indices is True and set labels to -100 where it's not masked

        outputs = self(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=labels
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        selfies = batch['selfies']
        batch_size = len(selfies)

        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        labels = tokenized_selfies['input_ids'].clone()
        mask_indices = torch.rand(tokenized_selfies['input_ids'].shape) < self.mask_pct
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.pad_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.eos_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.bos_token_id
        tokenized_selfies['input_ids'][mask_indices] = self.tokenizer.mask_token_id

        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels_copy = labels.clone()

        labels[~mask_indices] = -100        
        # Apply mask where mask_indices is True and set labels to -100 where it's not masked
        outputs = self(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=labels
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        
        # decode original selfies
        decoded = self.tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
        #decode predicted selfies
        predicted = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(decoded, predicted)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("val_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted selfies
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sf.decoder(sm)), 2, nBits=1024) for sm in decoded]
        fps_predicted = []
        errors = 0
        for sm in predicted:
            try:                
                mol = Chem.MolFromSmiles(sf.decoder(sm))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fps_predicted.append(fp)
                else:
                    fps_predicted.append(ExplicitBitVect(1024))
                    errors += 1
            except:
                fps_predicted.append(ExplicitBitVect(1024))
                errors += 1
        self.log('val_error_pct', errors / len(predicted), prog_bar=True, sync_dist=True, batch_size=batch_size)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        tanimoto = sum(tanimoto) / len(tanimoto)
        
        self.log("val_tanimoto", tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        selfies = batch['selfies']
        batch_size = len(selfies)
        
        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        labels = tokenized_selfies['input_ids'].clone()
        mask_indices = torch.rand(tokenized_selfies['input_ids'].shape) < self.mask_pct
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.pad_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.eos_token_id
        mask_indices &= tokenized_selfies['input_ids'] != self.tokenizer.bos_token_id
        tokenized_selfies['input_ids'][mask_indices] = self.tokenizer.mask_token_id

        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        labels_copy = labels.clone()

        labels[~mask_indices] = -100
        # Apply mask where mask_indices is True and set labels to -100 where it's not masked

        outputs = self(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            labels=labels
        )
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)

        
        
        # decode original selfies
        decoded = self.tokenizer.batch_decode(labels_copy, skip_special_tokens=True)
        #decode predicted selfies
        predicted = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(decoded, predicted)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("test_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted selfies
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sf.decoder(sm)), 2, nBits=1024) for sm in decoded]
        fps_predicted = []
        errors = 0
        for sm in predicted:
            try:                
                mol = Chem.MolFromSmiles(sf.decoder(sm))
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fps_predicted.append(fp)
                else:
                    fps_predicted.append(ExplicitBitVect(1024))
                    errors += 1
            except:
                fps_predicted.append(ExplicitBitVect(1024))
                errors += 1
        self.log('test_error_pct', errors / len(predicted), prog_bar=True, sync_dist=True, batch_size=batch_size)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        tanimoto = sum(tanimoto) / len(tanimoto)
        
        self.log("test_tanimoto", tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        
        return loss
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    


if __name__ == "__main__":


    # Initialize wandb
    wandb_logger = WandbLogger(
        project="", # project name
        entity="", # entity name
        # id='2ju60nmw',
        config=config,
        log_model=False,
        mode="online",
    )
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    # Load dataset
    dataset = MSDataModule(
        config["dataset_path"], 
        config["batch_size"],
    )

    # Initialize model
    model = MolBert(
        lr=config["lr"],
        mask_pct=config["mask_pct"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_loss", mode="min"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        # strategy="ddp",
        precision='16-mixed',
        sync_batchnorm=True,
        use_distributed_sampler=True,
        max_epochs=5,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.,
        logger=wandb_logger,
        accumulate_grad_batches=8,
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)
    # Evaluate model - load its checkpoint then test
    trainer.test(model, datamodule=dataset)

    # Finish wandb run
    wandb_logger.experiment.finish()