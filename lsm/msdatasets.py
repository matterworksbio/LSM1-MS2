import numpy as np
import zarr
from torch.utils.data import Dataset
import pandas as pd
import selfies as sf

# set constraints to 5 so I don't have to use different data
constraints = sf.get_semantic_constraints()
constraints["N"] = 5
sf.set_semantic_constraints(constraints)


"""
MSDataset Class:
    - Dataset class for loading MS data
    - Can be used for pretraining, spectral prediction, and property prediction, modified my 'mode' argument
    
    Args:
        dataset_path: path to zarr dataset
        seq_len: number of peaks to use per spectrum (default 64)
        mode: 'pretrain', 'spectral', or 'property' (default 'pretrain')
        tanimoto_path: path to tanimoto dataframe (default None), required if mode == 'spectral'
        train_minmax_path: path to train minmax dataframe (default None), required if mode == 'property'
        smiles_path: path to smiles dataframe (default None), required if mode == 'gen'
        
    
    Returns:
        if mode == 'pretrain':
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: zero vector
        if mode == 'spectral':
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: morgan molecular fingerprints of spectrum
        if mode == 'property':
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: property values of spectrum: 209 properties predicted from all RDKit descriptors        
        if mode == 'gen':
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: property feats
                smiles: smiles string of molecule
                selfies: selfies string of molecule
        if mode == 'linear_probing':
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: property values of spectrum: 209 properties predicted from all RDKit descriptors
                embedding: embedding of spectrum from linear probing model
                
"""


class MSDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        seq_len: int = 64,
        mode="pretrain",
        tanimoto_path=None,
        train_minmax_path=None,
        smiles_path=None,
        lsm_embeddings_path=None,
    ):
        super().__init__()
        # constants
        self.dataset_path = dataset_path
        self.seq_len = seq_len
        self.data = zarr.open(f"{dataset_path}", mode="r")
        self.total_spectra = self.data.shape[0]
        self.mode = mode
        print(f"Data is {self.data.shape} dimensions")

        assert mode in ["pretrain", "spectral", "property", "gen", "linear_probing"]

        # Mode specific additional adata
        if mode == "spectral":
            assert tanimoto_path != None
            self.tanimoto_df = pd.read_pickle(f"{tanimoto_path}")

        if mode == "property":
            assert train_minmax_path != None
            self.minmax_df = pd.read_csv(f"{train_minmax_path}")
            # scaling factors for y_feats normalization
            self.min = np.array(self.minmax_df.iloc[0, :].values, dtype=np.float32)
            self.max = np.array(self.minmax_df.loc[1, :].values, dtype=np.float32)

        if mode == "gen":
            assert smiles_path != None
            self.smiles_df = pd.read_csv(f"{smiles_path}")

        if mode == "linear_probing":
            assert lsm_embeddings_path != None
            assert train_minmax_path != None

            self.lsm_embeddings = zarr.open(f"{lsm_embeddings_path}", mode="r")
            self.minmax_df = pd.read_csv(f"{train_minmax_path}")
            # scaling factors for y_feats normalization
            self.min = np.array(self.minmax_df.iloc[0, :].values, dtype=np.float32)
            self.max = np.array(self.minmax_df.loc[1, :].values, dtype=np.float32)

    def __len__(self):
        return self.total_spectra

    @staticmethod
    def round_mz(mz, ppm=1.0):
        abs_mz_tol = (ppm * mz) / 1e6
        log_values = np.log10(abs_mz_tol + 1e-12)
        abs_log_values = np.abs(log_values)
        decimal_places = abs_log_values.astype("int16")
        vectorized_round = np.vectorize(lambda mz_val, dp: np.round(mz_val, dp))
        rounded_mz = vectorized_round(mz, decimal_places)
        return rounded_mz

    """
    process_spectra:
        - Processes spectra into dictionary of values, this is the workhorse of the class
        Args:
            spectra: spectra from zarr dataset, of shape [1, 4] or [1, 5] depending on mode ([1, 4] for pretrain, [1, 5] for spectral and property)
        
        Returns:
            Dictionary with keys:
                mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                precursormz: precursor m/z
                mz: m/z values of peaks
                inty: intensity values of peaks
                y_feats: zero vector (pretrain), morgan molecular fingerprints of spectrum (spectral), property values of spectrum (property)
        
    """

    def process_spectra(self, spectra):
        # Get mz, inty, mode, and rt
        mz = spectra[0]
        inty = spectra[1]
        precursormz = spectra[2]
        mode = spectra[3]
        if self.mode == "property" or self.mode == "linear_probing":
            y_feats = spectra[4]
            y_feats = (y_feats - self.min) / ((self.max - self.min) + 1e-6)
        elif self.mode == "spectral":
            y_feats = spectra[4]
        else:
            y_feats = np.zeros(10)

        # remove peaks with mz > 1000
        mz_indices = np.where(mz < 1000)[0]
        mz = mz[mz_indices]

        if mz.shape[0] == 0:
            mz = np.zeros_like(inty)
            inty = np.zeros_like(inty)
        else:
            inty = inty[mz_indices]
        # Randomly select num_peaks peaks (in case there are more than seq_len)
        num_peaks = min(self.seq_len, mz.shape[0])
        highest_inty_peaks = np.argsort(inty)[::-1][: self.seq_len + 128]
        peaks = np.random.choice(highest_inty_peaks, num_peaks, replace=False)
        peaks = np.sort(peaks)

        # Pad with zeros if there are less than seq_len peaks
        try:
            mz = self.round_mz(mz[peaks])
        except:
            print(spectra)
        mz = np.clip(mz, 0, 1_000)
        mz = np.pad(
            mz, (0, self.seq_len - mz.shape[0]), "constant", constant_values=0.0
        )

        # Pad with zeros if there are less than seq_len peaks
        inty = inty[peaks]
        inty = np.clip(inty, 0, 1_000)
        inty = np.pad(
            inty, (0, self.seq_len - inty.shape[0]), "constant", constant_values=0.0
        )

        # get index where padding starts
        try:
            pad_index = (np.where(mz == 0)[0][0]).astype(np.int8)
        except:
            pad_index = self.seq_len

        return (
            mode.astype(np.float32),
            precursormz.astype(np.float32),
            mz.astype(np.float32),
            inty.astype(np.float32),
            y_feats.astype(np.float32),
            pad_index,
        )

    """
    __getitem__:
        - Gets item from dataset
        Args:
            idx: index of item to get
        Returns:
            If mode is pretrain or property:
                Dictionary with keys:
                    mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                    precursormz: precursor m/z
                    mz: m/z values of peaks
                    inty: intensity values of peaks
                    y_feats: zero vector (pretrain),  property values of spectrum (property)
            If mode is spectral:
                for both idx and a random index in the dataset (for contrastive purposes):
                    Dictionary with keys: 
                        mode: 0 for positive spectra, 1 for negative spectra, -1 for unknown
                        precursormz: precursor m/z
                        mz: m/z values of peaks
                        inty: intensity values of peaks
                        y_feats: morgan molecular fingerprints of spectrum (spectral)
                    
    """

    def __getitem__(self, idx):
        # Each spectra is shape (4,) being the mz, intensity, mode, and rt representing the peaks
        spectra = self.data[idx]  # shape (4, )

        # Get mz, inty, mode, and rt
        mode, precursormz, mz, inty, y_feats, pad_idx = self.process_spectra(spectra)

        if self.mode == "property" or self.mode == "pretrain":
            return {
                "mode": mode,
                "precursormz": precursormz,
                "mz": mz,
                "inty": inty,
                "y_feats": y_feats,
                "pad_idx": pad_idx,
            }

        elif self.mode == "linear_probing":
            embedding = self.lsm_embeddings[idx]
            return {
                "mode": mode,
                "precursormz": precursormz,
                "mz": mz,
                "inty": inty,
                "y_feats": y_feats,
                "pad_idx": pad_idx,
                "embedding": embedding,
            }
        # for generative mode, we have to get inputs ready for chemgpt
        elif self.mode == "gen":
            smiles = self.smiles_df.iloc[idx]["smiles"]
            selfie = sf.encoder(smiles)  # convert to selfie
            return {
                "mode": mode,
                "precursormz": precursormz,
                "mz": mz,
                "inty": inty,
                "y_feats": y_feats,
                "pad_idx": pad_idx,
                "smiles": smiles,
                "selfies": selfie,
            }

        else:
            # randomly sample second 'contrastive' row
            idx2 = np.random.randint(0, self.total_spectra)

            # get fingerprints
            fingerprint = self.tanimoto_df.iloc[idx]["fingerprint"]
            smiles1 = self.tanimoto_df.iloc[idx]["smiles"]
            fingerprint2 = self.tanimoto_df.iloc[idx2]["fingerprint"]
            smiles2 = self.tanimoto_df.iloc[idx2]["smiles"]

            # get second spectrum
            spectrum2 = self.data[idx2]

            mode2, precursormz2, mz2, inty2, y_feats2, pad_idx2 = self.process_spectra(
                spectrum2
            )

            return {
                "mode": mode,
                "mode2": mode2,
                "precursormz": precursormz,
                "precursormz2": precursormz2,
                "mz": mz,
                "mz2": mz2,
                "inty": inty,
                "inty2": inty2,
                "y_feats": y_feats,
                "y_feats2": y_feats2,
                "fingerprint": fingerprint,
                "fingerprint2": fingerprint2,
                "pad_idx": pad_idx,
                "pad_idx2": pad_idx2,
                "smiles1": smiles1,
                "smiles2": smiles2,
            }


if __name__ == "__main__":
    dataset = MSDataset("path_to_data/ssl_train_data.zarr")

    print(dataset[0]["mz"].shape)
    print(dataset[0]["inty"].shape)
    print(dataset[0]["mode"].shape)
    print(dataset[0]["precursormz"].shape)
    print(dataset[0]["mz"])
    print(dataset[0]["inty"])
