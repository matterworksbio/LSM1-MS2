import torch
import torch.nn.functional as F
from einops import rearrange

#adapted into torch from https://github.com/matchms/matchms/blob/master/matchms/similarity/vector_similarity_functions.py#L94

def jaccard_index(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Computes the Jaccard-index (or Jaccard similarity coefficient) for batches of boolean
    1-D arrays.
    
    The Jaccard index between 2-D boolean arrays `u` and `v`,
    is defined as

    .. math::

       J(u,v) = \\frac{u \cap v}
                {u \cup v}

    Parameters
    ----------
    u :
        Input tensor of shape (batch_size, features). Expects boolean matrix.
    v :
        Input tensor of shape (batch_size, features). Expects boolean matrix.

    Returns
    -------
    jaccard_similarity
        The Jaccard similarity coefficient between vectors `u` and `v`. Shape (batch_size,).
    """
    
    u_or_v = torch.bitwise_or(u != 0, v != 0).sum(dim=1, dtype=torch.float16)
    u_and_v = torch.bitwise_and(u != 0, v != 0).sum(dim=1, dtype=torch.float16)

    jaccard_score = u_and_v / u_or_v
    jaccard_score[u_or_v == 0] = 0

    return jaccard_score

def cosine_similarity_matrix(tensor1, tensor2):
    # Normalize the rows of the tensors
    tensor1_reshaped = tensor1.unsqueeze(2)
    tensor2_reshaped = tensor2.unsqueeze(1)
    # Compute the cosine similarity matrix for each batch
    similarity_matrix = F.cosine_similarity(tensor1_reshaped, tensor2_reshaped, dim=3)
    
    flattened_similarity_matrix = rearrange(similarity_matrix, 'b i j -> b (i j)')
    return flattened_similarity_matrix
