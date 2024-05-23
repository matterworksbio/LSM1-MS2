import torch
import torch.nn as nn
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from msdatasets import MSDataset

# from train import SSModel
from pretrain_MAE import SSModel
from einops import rearrange
import numpy as np

# from train import SSModel
original_sys_path = sys.path.copy()
sys.path.append("path_to_this/lsm/hf_pretrain")
from hf_pretrain.pretrain_selfies_bert import MolBert
from hf_pretrain.pretrain_selfies_decoder import SelfiesDecoder
sys_path = sys.path

from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer
)
import os
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

import selfies as sf

torch.set_float32_matmul_precision("medium")

pl.seed_everything(42)

config = {
    "dataset_path": "/path_to_this/datasets/",
    "batch_size": 128,
    "input_dim": 1024,
    "lr": 2.5e-4,
    "temperature": 1.0,
    "inference": False,
    "inference_num_sequence": 100,
    "checkpoint_path": None#,
}

"""
class MSDataModule(pl.LightningDataModule):
    - simple datamodule for MS2 dataset
    - loads train, val, and test datasets
    
    Args:
        dataset_path: str - path to dataset
        batch_size: int - batch size
    
    Returns:
        train_dataloader: - returns train dataloader
        val_dataloader: - returns list of val dataloaders
        test_dataloader: - returns list of test dataloaders
    
    Methods:
        prepare_data: - no data preparation needed
        setup: - loads train, val, and test datasets
        train_dataloader: - returns train dataloader
        val_dataloader: - returns val dataloaders
        test_dataloader: - returns test dataloaders
    
"""


class MSDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set = MSDataset(
            f"{self.dataset_path}/train/no_casmi_train.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/no_casmi_train_smiles.csv",
        )
        self.valid_set = MSDataset(
            f"{self.dataset_path}/val/no_casmi_val.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/no_casmi_val_smiles.csv",
        )
        self.casmi = MSDataset(
            f"{self.dataset_path}/test/casmi2017.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/casmi2017_smiles.csv",
        )
        self.sampled_valid_set = MSDataset(
            f"{self.dataset_path}/val/disjoint_val.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/disjoint_val_smiles.csv",
        )

        # test data
        self.test_disjoint = MSDataset(
            f"{self.dataset_path}/test/disjoint_test.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/disjoint_test_smiles.csv",
        )
        self.test_casmi = MSDataset(
            f"{self.dataset_path}/test/casmi.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/casmi_smiles.csv",
        )
        self.test = MSDataset(
            f"{self.dataset_path}/test/test.zarr",
            mode="gen",
            smiles_path=f"{self.dataset_path}/smiles/test_smiles.csv",
        )

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
        loader1 = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=True,  # set shuffle to true, since running into r^2 collaps otherwise
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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        loader4 = DataLoader(
            self.test_casmi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return [loader1, loader2, loader3, loader4]

    def test_dataloader(self):
        loader1 = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        loader2 = DataLoader(
            self.test_casmi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        loader3 = DataLoader(
            self.test_disjoint,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        loader4 = DataLoader(
            self.casmi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )


        return [loader1, loader2, loader3, loader4]

"""
Class: Finetune_SSModel:
    - finetune the LSM model to align embeddings with the BERT model - then feeding in these embeddings into the pre-trained BERT model to generate SMILES strings
    - Decoder model is optionally trained as well
    - learns to a) align embeddings b) create embeddings which optimally reconstruct the input SMILES strings
    - loss functions are threefold: MSE (embeddings), Cross Entropy (smiles token logits), Cosine Similarity (embeddings)
    
    Args:
        input_dim: int - input dimension
        lr: float - learning rate 
        temperature: float - temperature for sampling
        inference: bool - inference mode (true if during inference)
        inference_num_sequence: int - number of sequences to generate during inference
    
    Logged metrics:
        train:
            - train_loss: total loss
            - train_CE_loss: cross entropy loss
            - train_mse_loss: mean squared error loss
            - train_cos_loss: cosine similarity loss
        val:
            - val_loss: total loss
            - val_CE_loss: cross entropy loss
            - val_mse_loss: mean squared error loss
            - val_cos_loss: cosine similarity loss
            - val_tanimoto: mean tanimoto similarity
            - val_cosine: mean cosine similarity
            - val_tanimoto_not_failed: mean tanimoto similarity for valid molecules 
            - val_cosine_not_failed: mean cosine similarity for valid molecules
            - val_failed_pct: percent of failed molecules
            - val_tanimoto_40: percent of tanimoto similarity above 0.4 (enveda meaningful match)
            - val_tanimoto_65: percent of tanimoto similarity above 0.65 (enveda close match)
            - val_tanimoto_100: percent of tanimoto similarity at 1.0 (exact match)
    
    Methods:
        init:
            - initialize the lsm from checkpoint, set mask_pct to 0.0 to disable masking
            - initialize the reshape layer to the output dimension
            - initialize the tokenizer and collator
            - initialize the encoder and decoder models (from pretrained models huggingface)
            - set gradients to false for encoder (sometimes also for for decoder since this improves performance)
            - initialize losses: cosine similarity, MSE, Cross Entropy
        forward:
            lsm:
                - pass inputs through lsm
                - extract cls token
                - pass through reshape layer
            pre-trained encoder:
                - pass inputs through encoder model
                - get hidden states (mean embeddings)
                
            decoder:
                - pass inputs through decoder model using lsm embeddings as input context
                - return logits, hidden states, z
            
            if mode is train:
                return decoder
                
            otherwise:  
                return decoder and reconstructed smiles (5 different sequences generated usign decoder)
        
        configure_optimizers:
            - return AdamW optimizer with learning rate (steady learning rate)
        
        training_step:
            - pass inputs through lsm and encoder
            - reshape logits (flatten)
            - calculate loss on logits, hidden states, and z
            - return loss
        validation_step:
            - pass inputs through lsm and encoder
            - reshape logits (flatten)
            - calculate loss on logits, hidden states, and z
            - calculate tanimoto and cosine similarity, and derived metrics
            - return loss
        test_step:
            - pass inputs through lsm and encoder
            - reshape logits (flatten)
            - calculate loss on logits, hidden states, and z
            - calculate tanimoto and cosine similarity, and derived metrics
            - return metrics dictionary
        
            
            
"""


class MS2Gen(pl.LightningModule):
    def __init__(self, input_dim, lr, temperature, inference=False, inference_num_sequence=100):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.lr = lr
        self.temperature = temperature
        self.num_sequence = 1 #only do one generated sequence for training
        if inference:
            self.num_sequence = inference_num_sequence
        
        
        print(f"learning reate is {self.lr}")

        # load lsm
        self.lsm_model = SSModel.load_from_checkpoint(
            f"" # insert path to pre lsm model checkpoint
        )
        # self.lsm_model = SSModel(d_model=1024, heads=16, depth=16, lr=self.lr)
        self.lsm_model = self.lsm_model.model.encoder
        self.lsm_model.mask_pct = 0.0  # set mask_pct to 0 to disable masking

        # load bert style model
        self.encoder_model = MolBert.load_from_checkpoint(
            "path_to_this/selfies_bert_best.ckpt"
        )
        self.encoder_model.mask_pct = 0.0
        self.encoder_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "zjunlp/MolGen-large", max_len=256
        )
        self.collator = DataCollatorWithPadding(
            self.tokenizer, padding=True, return_tensors="pt"
        )

        self.decoder_model = SelfiesDecoder.load_from_checkpoint('path_to_this/gpt_decoder_selfies_best.ckpt')
        self.decoder_model = self.decoder_model.decoder_model

        # set gradients to false for encoder (enable for decoder since this improves performance)
        for param in self.encoder_model.parameters():
            param.requires_grad = False

        # losses
        self.cosine_similarity = nn.CosineEmbeddingLoss(
            margin=0.0
        )  # margin of 0.0 since no contrastive
        self.MSE = nn.MSELoss()
        self.Cross_Entropy = nn.CrossEntropyLoss()

    def forward(self, mz, inty, precursor_mz, bert_inputs, smiles, mode="train"):
        # pass inputs through lsm
        z, _ = self.lsm_model(precursor_mz, mz, inty)
        z = z[:, 0, :]  # extract cls token
        z = z.unsqueeze(1)
        
        # inference mode
        if mode == "inference":
            decoder_inputs = torch.tensor(
                [[self.tokenizer.bos_token_id] for i in range(len(smiles))]
            ).to(z.device)

            gen = self.decoder_model.generate(
                decoder_inputs,  # take just bos token for generative task
                encoder_hidden_states=z,
                do_sample=True,  
                max_length=256,
                temperature=self.temperature,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=self.num_sequence,
            )
            reconstructed_selfies = self.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )
            return reconstructed_selfies

        #otherwise if we are training the model

        # pass tokenized inputs through encoder model
        outputs = self.encoder_model(**bert_inputs, output_hidden_states=True)
        full_embeddings = outputs[1][-1]
        mask = bert_inputs["attention_mask"]
        mean_embeddings = (full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            -1
        ).unsqueeze(-1)
        hidden_states = mean_embeddings[
            :, None
        ]  # hidden states shape (bs, 1, -1) - recommended for decoder model

        decoder_tokens = bert_inputs["input_ids"][
            :, :-1
        ]  # Exclude the last token for each sequence

        logits = self.decoder_model(
            input_ids=decoder_tokens,  # take all tokens except last for training step
            encoder_hidden_states=z,
        ).logits

        if mode == "train":
            return logits, hidden_states, z
        else:
            decoder_inputs = torch.tensor(
                [[self.tokenizer.bos_token_id] for i in range(len(smiles))]
            ).to(z.device)

            gen = self.decoder_model.generate(
                decoder_inputs,  # take just bos token for generative task
                encoder_hidden_states=z,
                do_sample=False,
                num_beams=100,  
                # top_k=10,
                max_length=256,
                temperature=self.temperature,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=self.num_sequence,
            )
            reconstructed_smiles = self.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )
            return logits, hidden_states, z, reconstructed_smiles

    def training_step(self, batch, batch_idx):
        mz, inty, precursormz, selfies = (
            batch["mz"],
            batch["inty"],
            batch["precursormz"],
            batch["selfies"],
        )
        bert_inputs = self.collator(self.tokenizer(selfies))
        bert_inputs = {k: v.cuda() for k, v in bert_inputs.items()}

        logits, hidden_states, z = self(
            mz, inty, precursormz, bert_inputs, selfies, mode="train"
        )

        # reshape logits (flatten)
        logits = rearrange(logits, 'b l v -> (b l) v')

        # Prepare labels, shifting and flattening using einops, and mask padding positions
        labels = rearrange(bert_inputs["input_ids"][:, 1:], 'b l -> (b l)')
        attention_mask = rearrange(bert_inputs["attention_mask"][:, 1:], 'b l -> (b l)')
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)
        MSE_Loss = self.MSE(hidden_states, z)

        hidden_states = hidden_states.squeeze(1)
        z = z.squeeze(1)

        target = torch.ones(hidden_states.size(0)).to(
            hidden_states.device
        )  # target is 1 to maximize similarity and ensure target is on the same device
        cosine_sim_loss = self.cosine_similarity(hidden_states, z, target)

        loss = MSE_Loss + cosine_sim_loss + CE_loss

        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log(
            "train_CE_loss",
            CE_loss,
            sync_dist=True,
        )
        self.log(
            "train_mse_loss",
            MSE_Loss,
            sync_dist=True,
        )
        self.log(
            "train_Cos_loss",
            cosine_sim_loss,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mz, inty, precursormz, selfies = (
            batch["mz"],
            batch["inty"],
            batch["precursormz"],
            batch["selfies"],
        )
        batch_size = len(selfies)
        bert_inputs = self.collator(self.tokenizer(selfies))
        bert_inputs = {
            k: v.cuda() for k, v in bert_inputs.items()
        }  # move to device since default dict is on cpu

        logits, hidden_states, z, reconstructed = self(
            mz, inty, precursormz, bert_inputs, selfies, mode="val"
        )
        # print types and devices on all inputs and outputs

        # reshape logits (flatten)
        logits = rearrange(logits, 'b l v -> (b l) v')

        # Prepare labels, shifting and flattening using einops, and mask padding positions
        labels = rearrange(bert_inputs["input_ids"][:, 1:], 'b l -> (b l)')
        attention_mask = rearrange(bert_inputs["attention_mask"][:, 1:], 'b l -> (b l)')
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)
        MSE_Loss = self.MSE(hidden_states, z)

        hidden_states = hidden_states.squeeze(1)
        z = z.squeeze(1)

        target = torch.ones(hidden_states.size(0)).to(
            hidden_states.device
        )  # target is 1 to maximize similarity and ensure target is on the same device
        cosine_sim_loss = self.cosine_similarity(hidden_states, z, target)
        
        loss = MSE_Loss + CE_loss + cosine_sim_loss

        # log losses
        self.log("val_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "val_CE_loss",
            CE_loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_mse_loss",
            MSE_Loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_cos_loss",
            cosine_sim_loss,
            sync_dist=True,
            batch_size=batch_size,
        )

        # turn smiles/selfies into fingerprint and compare tanimoto similarity and cosine similarity
        # recreate reconstructed smiles as fingerprints, if the molecules are not valid, impute with zeros
        # decode original selfies
        reconstructed_smiles = [sf.decoder(sm) for sm in reconstructed]
        smiles = [sf.decoder(sm) for sm in selfies]
        
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(smiles, reconstructed_smiles)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("val_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted smiles
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sm), 2, nBits=1024) for sm in smiles]
        fps_predicted = []
        errors = 0
        for sm in reconstructed_smiles:
            mol = AllChem.MolFromSmiles(sm)
            if mol is not None:
                fps_predicted.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            else:
                errors += 1
                fps_predicted.append(ExplicitBitVect(1024))
        self.log('val_error_pct', errors / len(reconstructed), prog_bar=True, sync_dist=True)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_tanimoto = sum(tanimoto) / len(tanimoto)
        tanimoto_median = np.median(tanimoto)
        tanimoto_over675 = sum([1 for i in tanimoto if i > 0.675]) / len(tanimoto)
        tanimoto_over40 = sum([1 for i in tanimoto if i > 0.4]) / len(tanimoto)
        tanimoto_100 = sum([1 for i in tanimoto if i == 1.0]) / len(tanimoto)
        tanimoto_over95 = sum([1 for i in tanimoto if i > 0.95]) / len(tanimoto)
        
        self.log("val_tanimoto", avg_tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_median", tanimoto_median, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_675", tanimoto_over675, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_40", tanimoto_over40, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_100", tanimoto_100, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_95", tanimoto_over95, prog_bar=True, sync_dist=True, batch_size=batch_size)

        #cosine similarity of the fingerprints
        cosine = [AllChem.DataStructs.FingerprintSimilarity(fp1, fp2, metric=AllChem.DataStructs.CosineSimilarity) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_cosine = sum(cosine) / len(cosine)
        self.log("val_cosine", avg_cosine, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        mz, inty, precursormz, selfies = (
            batch["mz"],
            batch["inty"],
            batch["precursormz"],
            batch["selfies"],
        )
        batch_size = len(selfies)
        bert_inputs = self.collator(self.tokenizer(selfies))
        bert_inputs = {
            k: v.cuda() for k, v in bert_inputs.items()
        }  # move to device since default dict is on cpu

        logits, hidden_states, z, reconstructed = self(
            mz, inty, precursormz, bert_inputs, selfies, mode="val"
        )
        # print types and devices on all inputs and outputs

        # reshape logits (flatten)
        logits = rearrange(logits, 'b l v -> (b l) v')

        # Prepare labels, shifting and flattening using einops, and mask padding positions
        labels = rearrange(bert_inputs["input_ids"][:, 1:], 'b l -> (b l)')
        attention_mask = rearrange(bert_inputs["attention_mask"][:, 1:], 'b l -> (b l)')
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)
        MSE_Loss = self.MSE(hidden_states, z)

        hidden_states = hidden_states.squeeze(1)
        z = z.squeeze(1)

        target = torch.ones(hidden_states.size(0)).to(
            hidden_states.device
        )  # target is 1 to maximize similarity and ensure target is on the same device
        cosine_sim_loss = self.cosine_similarity(hidden_states, z, target)
        
        loss = MSE_Loss + CE_loss + cosine_sim_loss

        # log losses
        self.log("test_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "test_CE_loss",
            CE_loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test_mse_loss",
            MSE_Loss,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "test_cos_loss",
            cosine_sim_loss,
            sync_dist=True,
            batch_size=batch_size,
        )

        # turn smiles/selfies into fingerprint and compare tanimoto similarity and cosine similarity
        # recreate reconstructed smiles as fingerprints, if the molecules are not valid, impute with zeros
        # decode original selfies
        reconstructed_smiles = [sf.decoder(sm) for sm in reconstructed]
        smiles = [sf.decoder(sm) for sm in selfies]
        
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(smiles, reconstructed_smiles)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("val_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted smiles
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sm), 2, nBits=1024) for sm in smiles]
        fps_predicted = []
        errors = 0
        for sm in reconstructed_smiles:
            mol = AllChem.MolFromSmiles(sm)
            if mol is not None:
                fps_predicted.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            else:
                errors += 1
                fps_predicted.append(ExplicitBitVect(1024))
        self.log('val_error_pct', errors / len(reconstructed), prog_bar=True, sync_dist=True)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_tanimoto = sum(tanimoto) / len(tanimoto)
        tanimoto_median = np.median(tanimoto)
        tanimoto_over675 = sum([1 for i in tanimoto if i > 0.675]) / len(tanimoto)
        tanimoto_over40 = sum([1 for i in tanimoto if i > 0.4]) / len(tanimoto)
        tanimoto_100 = sum([1 for i in tanimoto if i == 1.0]) / len(tanimoto)
        tanimoto_over95 = sum([1 for i in tanimoto if i > 0.95]) / len(tanimoto)
        
        self.log("test_tanimoto", avg_tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_median", tanimoto_median, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_675", tanimoto_over675, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_40", tanimoto_over40, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_100", tanimoto_100, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_95", tanimoto_over95, prog_bar=True, sync_dist=True, batch_size=batch_size)

        #cosine similarity of the fingerprints
        cosine = [AllChem.DataStructs.FingerprintSimilarity(fp1, fp2, metric=AllChem.DataStructs.CosineSimilarity) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_cosine = sum(cosine) / len(cosine)
        self.log("test_cosine", avg_cosine, prog_bar=True, sync_dist=True)


        return {
            "loss": loss,
            "tanimoto": sum(tanimoto) / len(tanimoto),
            "mse_loss": MSE_Loss,
            "ce_loss": CE_loss,
            "cos_loss": cosine_sim_loss,
            "tanimoto_40": tanimoto_over40,
            "tanimoto_65": tanimoto_over675,
            "tanimoto_95": tanimoto_over95,
            "tanimoto_100": tanimoto_100,
            "cosine": avg_cosine,
            "perfect_recon": perfect_recon,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    pl.seed_everything(42)

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="",
        entity="",
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
    )

    # Initialize model
    model = MS2Gen(
        config["input_dim"],
        config["lr"],
        config["temperature"],
        config["inference"],
        config["inference_num_sequence"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="trained_models",
        filename=f"{model_name}_best",
        monitor="val_tanimoto/dataloader_idx_0", # max tanimoto on casmi
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        max_epochs=30,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.0,
        logger=wandb_logger,
        accumulate_grad_batches=1,
    )

    if config["checkpoint_path"] is not None:
        print("Loading pre-trained checkpoint")
            # Evaluate on test dataset
        model = MS2Gen.load_from_checkpoint(config["checkpoint_path"])
        # trainer.fit(model, dataset, ckpt_path=config["checkpoint_path"])
    else:
        trainer.fit(model, dataset)

    if os.path.exists(f'./trained_models/{model_name}_best.ckpt'):
        model = MS2Gen.load_from_checkpoint(f'./trained_models/{model_name}_best.ckpt')
        # Evaluate on test dataset
        test_results = trainer.test(model, datamodule=dataset)
        print(f"Test Results: {test_results}")

    # Finish wandb run
    wandb_logger.experiment.finish()
