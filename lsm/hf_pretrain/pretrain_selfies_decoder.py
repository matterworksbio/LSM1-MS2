import torch
import torch.nn as nn
import pandas as pd
import sys
import pytorch_lightning as pl
import torch.utils.data as data
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from hf_pretrain.pretrain_selfies_bert import MolBert
import numpy as np
from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer,
    GPT2Config,
)
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import selfies as sf

# from train import SSModel
original_sys_path = sys.path.copy()
sys.path.append("") #path to lsm folder
from models.conditional_gpt2_model import ConditionalGPT2LMHeadModel

sys_path = sys.path


torch.set_float32_matmul_precision("medium")

config = {
    "dataset_path": "/datasets/hf_smiles/zinc100m.parquet", # path to parquet file of SMILES/Selfies
    "batch_size": 192,
    "input_dim": 1024,
    "heads": 8,
    "depth": 6,
    "lr": 1e-4,
    "tokenizer_max_len": 256,
    "checkpoint_path": None,  #',
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


class SMILESDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
    ):
        self.df = pd.read_parquet(dataset_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"selfies": self.df.iloc[idx]["SELFIES"]}


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
        self.train_set, val_test_set = data.random_split(
            all_data, [train_set_size, val_test_set_size]
        )
        self.valid_set, self.test_set = data.random_split(
            val_test_set, [val_set_size, test_set_size]
        )

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
            shuffle=False,  # set shuffle to true, since running into r^2 collaps otherwise
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


class SelfiesDecoder(pl.LightningModule):
    def __init__(self, input_dim, heads, num_layers, tokenizer_max_len, lr):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.heads = heads
        self.num_layers = num_layers
        self.tokenizer_max_len = tokenizer_max_len
        self.lr = lr
        print(f"learning rate is {self.lr}")

        # load bert style model
        self.encoder_model = MolBert.load_from_checkpoint(
            "hf_pretrain/trained_models/selfies_bert_best.ckpt"
        )
        self.encoder_model.mask_pct = 0.0
        self.encoder_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "zjunlp/MolGen-large", max_len=256
        )
        self.collator = DataCollatorWithPadding(
            self.tokenizer, padding=True, return_tensors="pt"
        )

        gpt_config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=self.tokenizer_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            n_layer=self.num_layers,
            n_head=self.heads,
            add_cross_attention=True,
            n_embd=self.input_dim,
        )

        self.decoder_model = ConditionalGPT2LMHeadModel(gpt_config)

        # set gradients to false for encoder (enable for decoder since this improves performance)
        for param in self.encoder_model.parameters():
            param.requires_grad = False

        self.Cross_Entropy = nn.CrossEntropyLoss()

    def forward(self, input_ids, selfies, attention_mask=None, mode="train"):
        # pass tokenized inputs through encoder model
        outputs = self.encoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        full_embeddings = outputs[1][-1]
        mask = attention_mask
        mean_embeddings = (full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(
            -1
        ).unsqueeze(-1)
        hidden_states = mean_embeddings[
            :, None
        ]  # hidden states shape (bs, 1, -1) - recommended for decoder model

        decoder_tokens = input_ids[:, :-1]  # Exclude the last token for each sequence

        logits = self.decoder_model(
            input_ids=decoder_tokens,  # take all tokens except last for training step
            encoder_hidden_states=hidden_states,
        ).logits

        if mode == "train":
            return logits, hidden_states
        else:
            decoder_inputs = torch.tensor(
                [[self.tokenizer.bos_token_id] for i in range(len(selfies))]
            ).to(hidden_states.device)

            gen = self.decoder_model.generate(
                decoder_inputs,  # take just bos token for generative task
                encoder_hidden_states=hidden_states,
                do_sample=True,  # greedy decoding is recommended
                max_length=256,
                temperature=1.0,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # make 5 different sequences
                # num_return_sequences=5,
            )
            reconstructed_selfies = self.tokenizer.batch_decode(
                gen, skip_special_tokens=True
            )
            return logits, hidden_states, reconstructed_selfies

    def training_step(self, batch, batch_idx):
        selfies = batch["selfies"]
        batch_size = len(selfies)

        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}


        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
                
        logits, hidden_states = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            selfies=selfies,
            mode="train",
        )
        labels = rearrange(inputs["input_ids"][:, 1:], "b l -> (b l)")
        attention_mask = rearrange(attention_mask[:, 1:], "b l -> (b l)")
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # reshape logits (flatten)
        logits = rearrange(logits, "b l v -> (b l) v")

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)

        loss = CE_loss

        self.log("train_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        selfies = batch["selfies"]
        batch_size = len(selfies)

        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
        
        logits, hidden_states, predicted = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            selfies=selfies,
            mode="val",
        )
        labels = rearrange(inputs["input_ids"][:, 1:], "b l -> (b l)")
        attention_mask = rearrange(attention_mask[:, 1:], "b l -> (b l)")
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # reshape logits (flatten)
        logits = rearrange(logits, "b l v -> (b l) v")

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)

        loss = CE_loss

        self.log("val_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

        # decode original selfies
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        print(decoded)
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(decoded, predicted)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("val_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted selfies
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sf.decoder(sm)), 2, nBits=1024) for sm in decoded]
        fps_predicted = []
        errors = 0
        for sm in predicted:
            mol = Chem.MolFromSmiles(sf.decoder(sm))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps_predicted.append(fp)
            else:
                fps_predicted.append(ExplicitBitVect(1024))
                errors += 1
        self.log('val_error_pct', errors / len(predicted), prog_bar=True, sync_dist=True, batch_size=batch_size)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_tanimoto = sum(tanimoto) / len(tanimoto)
        tanimoto_median = np.median(tanimoto)
        tanimoto_over60 = sum([1 for i in tanimoto if i > 0.6]) / len(tanimoto)
        
        self.log("val_tanimoto", avg_tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_median", tanimoto_median, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_tanimoto_over60", tanimoto_over60, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        selfies = batch["selfies"]
        batch_size = len(selfies)

        tokenized_selfies = self.tokenizer(
            selfies,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        inputs = self.collator(tokenized_selfies)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
        
        logits, hidden_states, predicted = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            selfies=selfies,
            mode="val",
        )
        labels = rearrange(inputs["input_ids"][:, 1:], "b l -> (b l)")
        attention_mask = rearrange(attention_mask[:, 1:], "b l -> (b l)")
        labels[attention_mask == 0] = -100  # Ignore padding in loss calculation

        # reshape logits (flatten)
        logits = rearrange(logits, "b l v -> (b l) v")

        # calculate losses
        CE_loss = self.Cross_Entropy(logits, labels)

        loss = CE_loss

        self.log("test_loss", loss, sync_dist=True, prog_bar=True, batch_size=batch_size)

        # decode original selfies
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
        # calculate accuracy (100%) string similarity
        perfect_recon = [1 if a == b else 0 for a, b in zip(decoded, predicted)]
        perfect_recon = sum(perfect_recon) / len(perfect_recon)
        
        self.log("test_perfect_recon", perfect_recon, prog_bar=True, sync_dist=True, batch_size=batch_size)
        
        # calculate tanimoto similarity of the decoded and predicted selfies
        fps_decoded = [AllChem.GetMorganFingerprintAsBitVect(AllChem.MolFromSmiles(sf.decoder(sm)), 2, nBits=1024) for sm in decoded]
        fps_predicted = []
        errors = 0
        for sm in predicted:
            mol = Chem.MolFromSmiles(sf.decoder(sm))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fps_predicted.append(fp)
            else:
                fps_predicted.append(ExplicitBitVect(1024))
                errors += 1
        self.log('test_error_pct', errors / len(predicted), prog_bar=True, sync_dist=True, batch_size=batch_size)
        tanimoto = [AllChem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp1, fp2 in zip(fps_decoded, fps_predicted)]
        avg_tanimoto = sum(tanimoto) / len(tanimoto)
        tanimoto_median = np.median(tanimoto)
        tanimoto_over60 = sum([1 for i in tanimoto if i > 0.6]) / len(tanimoto)
        
        self.log("test_tanimoto", avg_tanimoto, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_median", tanimoto_median, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("test_tanimoto_over60", tanimoto_over60, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    pl.seed_everything(42)

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="", #project name
        entity="", #project entity
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
    model = SelfiesDecoder(
        config["input_dim"],
        config["heads"],
        config["depth"],
        config["tokenizer_max_len"],
        config["lr"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="trained_models",
        filename=f"{model_name}_best",
        monitor="val_tanimoto",  # max tanimoto on casmi
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)

    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy="ddp_find_unused_parameters_true",
        precision="16-mixed",
        sync_batchnorm=True,
        use_distributed_sampler=True,
        max_epochs=10,
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.0,
        logger=wandb_logger,
        accumulate_grad_batches=4,
    )

    if config["checkpoint_path"] is not None:
        print("Loading pre-trained checkpoint")
        # Evaluate on test dataset
        model = SelfiesDecoder.load_from_checkpoint(config["checkpoint_path"])
        # trainer.fit(model, dataset, ckpt_path=config["checkpoint_path"])
    else:
        trainer.fit(model, dataset)

    if os.path.exists(f"./trained_models/{model_name}_best.ckpt"):
        model = SelfiesDecoder.load_from_checkpoint(
            f"./trained_models/{model_name}_best.ckpt"
        )
        # Evaluate on test dataset
        test_results = trainer.test(model, datamodule=dataset)
        print(f"Test Results: {test_results}")

    # Finish wandb run
    wandb_logger.experiment.finish()
