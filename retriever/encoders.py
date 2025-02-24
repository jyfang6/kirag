import torch 
import torch.nn as nn 
from torch import Tensor
import torch.nn.functional as F
from transformers import BertModel


class BertEncoder(BertModel):

    def __init__(self, config, project_dim: int = -1, pool_type: str = "cls", **kwargs):
        
        super().__init__(config)
        assert pool_type in ["cls", "pool", "none", "average"]
        if project_dim > 0:
            self.projection = nn.Linear(config.hidden_size, project_dim) if project_dim > 0 else None
            self.norm = nn.LayerNorm(project_dim)
        self.embedding_size = project_dim if project_dim > 0 else config.hidden_size 
        self.pool_type = pool_type
        self.kwargs = kwargs 

        self.init_weights()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
            
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        token_embed = last_hidden_states

        if hasattr(self, "projection"):
            token_embed = self.norm(self.projection(token_embed))

        if self.pool_type == "cls":
            out = (token_embed[:, 0], last_hidden_states)

        elif self.pool_type == "pool" or self.pool_type == "average":
            extended_attention_mask = attention_mask.unsqueeze(-1)
            masked_token_embed = token_embed * extended_attention_mask
            token_length = extended_attention_mask.sum(1)
            pool_token_embed = masked_token_embed.sum(1) / torch.where(token_length==0, 1, token_length)
            out = (pool_token_embed, last_hidden_states)
        
        elif self.pool_type == "none":
            out = (token_embed, last_hidden_states)

        else:
            raise NotImplemented(f"{self.pool_type} is not implemented!")
        
        return out


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5Encoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        embeddings = average_pool(last_hidden_states, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    

class ContrieverEncoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        last_hidden_states = transformer_output.last_hidden_state
        last_hidden_states = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden_states.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        return embeddings
    

class BGEEncoder(BertModel):

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, add_pooling_layer)
        self.kwargs = kwargs
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):

        transformer_output = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
        
        last_hidden_states = transformer_output.last_hidden_state
        embeddings = last_hidden_states[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
