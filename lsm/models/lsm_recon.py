import torch
from torch import nn
from einops import repeat
from x_transformers import ContinuousTransformerWrapper, Encoder 

'''
LSMEncoder Class:
    - Encoder class for LSM model
    - Uses ALiBi positional embedding
    - Uses continuous transformer wrapper
    
    Args:
        d_model: dimension of latent vector in model
        heads: number of transformer heads
        depth: number of transformer layers
        mask_pct: percent of peaks to mask out
    
    Returns:
        x: encoded peaks passed through transformer
        mask: bool array of peaks that were masked out
'''
class LSMEncoder(nn.Module):
    def __init__(self, d_model, heads, depth, mask_pct, ms2prop):
        super().__init__()
        # Constants
        self.mask_pct = mask_pct
        self.ms2prop = ms2prop
        

        if self.ms2prop == False:
            # Embedding layers
            self.mz_enc_1 = nn.Embedding(1_000, 128)
            self.mz_enc_2 = nn.Embedding(1_000, 128)
            self.inty_enc = nn.Embedding(2_001, 128)
            self.emb_peak = nn.Linear(128*3, d_model)
        else:
            print("Using MS2Prop")
            self.mz_enc = nn.Embedding(10_001, 512)
            self.emb_peak = nn.Linear(512+1, d_model)

        # Transformer
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len= 512 + 1,
            attn_layers=Encoder(
                dim=d_model,
                depth=depth,
                heads=heads,
                alibi_pos_bias = True, # turns on ALiBi positional embedding
                alibi_num_heads = 16    # only use ALiBi for 32 out of the 32 heads, 
            ),
            use_abs_pos_emb = False,
            post_emb_norm=True,
        )

        # Mask token (learned)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, precursormz, mz, inty):
        ## for reference: num_peak = num_tokens, each peak is a separate token
        
        # precursormz: (b, 1)
        # mz: (b, num_peaks)
        # inty: (b, num_peaks)
        
        b, p = mz.shape

        # Create an attention mask with False values where mz is 0 and True where mz is not 0
        attn_mask = (mz != 0)
        # Add anther element to account for mode
        attn_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mz.device), attn_mask], dim=1)
        mask = None
        
        if self.ms2prop == False:
            # Embed mz, max value is 1000
            mz = torch.round((mz / 1_000) * (1e6 - 1)).long()
            mz_1 = self.mz_enc_1(mz // 1000) # First 3 digits
            mz_2 = self.mz_enc_2(mz % 1000) # Last 3 digits

            inty = self.inty_enc(inty.long())

            # Embed precursormz, mz, and inty
            precursormz = precursormz.long()
            premz1 = self.mz_enc_1(precursormz // 1000) # First 3 digits
            premz2 = self.mz_enc_2(precursormz % 1000) # Last 3 digits
            preinty = self.inty_enc(torch.full_like(precursormz, 2000.)) #embed precursormz with inty = 2000, so it is different from other inty embeddings
            premz = torch.cat([premz1, premz2, preinty], dim=-1)
            premz = self.emb_peak(premz)
            
            # Embed peak
            peak = self.emb_peak(torch.cat([mz_1, mz_2, inty], dim=-1)) # (b, num_peaks, d_model)

            # Randomly replace mask_pct of the peak embeddings with the mask token
            mask = torch.rand((b, p), device=peak.device) < self.mask_pct
            mask = repeat(mask, 'b n -> b n e', e=peak.shape[-1])
            mask = mask.any(dim=-1)
            peak = torch.where(mask[:, :, None], self.mask_token, peak)
            
            # Concatenate precursor and peak
            x = torch.cat([premz, peak], dim=1) # (b, num_peaks + 1, d_model)
        
        else:
            #round mz to nearest 0.1 and discretize
            mz = torch.round(mz * 10).long()
            mz = self.mz_enc(mz)
            inty = inty.long() / 1000
            inty = inty.unsqueeze(-1)
            peak = torch.cat([mz, inty], dim=-1)
            peak = self.emb_peak(peak)
            
            precursormz = precursormz.long()
            premz = self.mz_enc(precursormz)
            preinty = torch.full_like(precursormz, 2.)
            preinty = preinty.unsqueeze(-1)
            precursor_peak = self.emb_peak(torch.cat([premz, preinty], dim=-1))
            
            #concatenate peaks tokether with precursor peak
            x = torch.cat([precursor_peak, peak], dim=1)
        
        x = self.transformer(x, mask=attn_mask)
        return x, mask
 
'''
class LSM Class:
    - LSM model class
    - Uses LSMEncoder class
    - Has 3 reconstruction projection heads for mz1, mz2, and inty
    
    Args:
        d_model: dimension of latent vector in model
        heads: number of transformer heads
        depth: number of transformer layers
        mask_pct: percent of peaks to mask out
'''
class LSM(nn.Module):
    def __init__(self, d_model=512, heads=8, depth=8, mask_pct=0.25, ms2prop=False):
        super().__init__()
        #NN layers
        self.encoder = LSMEncoder(d_model, heads, depth, mask_pct, ms2prop=ms2prop)
        self.projection_mz1 = nn.Sequential(
            nn.Linear(d_model, 1_000),
            nn.ReLU(),
        )
        self.projection_mz2 = nn.Sequential(
            nn.Linear(d_model, 1_000),
            nn.ReLU(),
        )
        self.projection_inty = nn.Sequential(
            nn.Linear(d_model, 1_001),
            nn.ReLU(),
        )

    def forward(self, precursor, mz, inty):
        emb , mask= self.encoder(precursor, mz, inty)
        emb = emb[:, 1:, :] #ignore precursormz token
        
        mz1_logits = self.projection_mz1(emb)
        mz2_logits = self.projection_mz2(emb)
        inty_logits = self.projection_inty(emb)
        return mz1_logits, mz2_logits, inty_logits, mask
