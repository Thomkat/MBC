import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25, decay: float = 0.99):
        # num_codes: Number of codes in the codebook
        # code_dim: Dimension of each code vector
        # commitment_cost: Weight for the commitment loss
        # decay: Decay factor for the EMA usage tracking

        super().__init__()
        
        # initialize codebook
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1/num_codes, 1/num_codes)

        self.commitment_cost = commitment_cost
        self.decay = decay
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))

    def forward(self, inputs: torch.Tensor):
        # flatten
        flat = inputs.view(-1, inputs.size(-1))
        
        # compute L2 distance to codebook
        dists = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        
        # nearest code
        codes = torch.argmin(dists, dim=1).unsqueeze(1)
        one_hot = torch.zeros(flat.size(0), self.embedding.num_embeddings, device=flat.device)
        one_hot.scatter_(1, codes, 1)
        quantized_flat = one_hot @ self.embedding.weight
        quantized = quantized_flat.view_as(inputs)

        # losses
        e_loss = F.mse_loss(quantized.detach(), inputs)
        q_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_loss + self.commitment_cost * e_loss

        # straight‑through estimator (STE)
        quantized = inputs + (quantized - inputs).detach()

        # perplexity
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Codebook resetting
        if self.training:
            with torch.no_grad():
                # Update the EMA (of code usage)
                self.ema_cluster_size.data.mul_(self.decay).add_(
                    one_hot.sum(0), alpha=1 - self.decay
                )

                # Find codes with usage count below threshold
                dead_codes = torch.where(self.ema_cluster_size < 1e-4)[0]
                num_dead = len(dead_codes)

                if num_dead > 0:
                    # Get random input vectors from the batch to be new codes
                    num_replacements = min(num_dead, flat.size(0))
                    
                    replacement_indices = torch.randperm(flat.size(0))[:num_replacements]
                    replacements = flat[replacement_indices]

                    # Assign the replacement vectors to the dead code slots
                    self.embedding.weight.data[dead_codes[:num_replacements]] = replacements.to(self.embedding.weight.dtype)

                    # Reset the usage count for the newly replaced codes to a higher value
                    self.ema_cluster_size.data[dead_codes[:num_replacements]] = self.ema_cluster_size.mean()

        return quantized, loss, perplexity, codes