import os
from accelerate.logging import get_logger
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

logger = get_logger(__name__)


class GenHPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.input2emb_model = TextEmb(args)
        if args.eventencoder == "transformer":
            self.eventencoder_model = TransformerEventEncoder(args)
        else:
            raise NotImplementedError
        self.pred_model = Aggregator(args)
        self.emb2out_model = PredOutPutLayer(args)

    def get_logits(self, net_output):
        return net_output['pred_output']

    def get_targets(self, sample):
        return sample["labels"]

    def forward(self, **kwargs):
        all_codes_embs = self.input2emb_model(**kwargs)  # (B, S, E)
        events = self.eventencoder_model(all_codes_embs, **kwargs)
        x = self.pred_model(events, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        return net_output


class TextEmb(nn.Module):
    def __init__(self, args, embed_dim=None):
        super().__init__()
        
        self.args = args
        
        self.input_index_size = 28996 # bio clinical bert vocab
        self.type_index_size = 7 
        self.dpe_index_size = 16

        self.dpe = args.dpe
        self.token_type = args.type_token
        self.pos_enc = args.pos_enc

        if embed_dim:
            self.args.embed_dim = embed_dim        

        self.input_ids_embedding = nn.Embedding(
            self.input_index_size, self.args.embed_dim, padding_idx=0
        )

        self.type_ids_embedding =nn.Embedding(
            self.type_index_size, self.args.embed_dim, padding_idx=0
        ) if self.args.type_token else None

        self.dpe_ids_embedding =nn.Embedding(
            self.dpe_index_size, self.args.embed_dim
        ) if self.args.dpe else None

        max_len = args.max_word_len
        
        self.pos_encoder = PositionalEncoding(  
            args.embed_dim, args.dropout, max_len
            ) if self.pos_enc else None
        
        self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

    def forward(self, input_ids, type_ids, dpe_ids, **kwargs):
        B, S = input_ids.shape[0], input_ids.shape[1] # time: hi - (B, S, 1), fl - (B, S, 1)
        
        x = self.input_ids_embedding(input_ids)

        if self.type_ids_embedding: # column description mean 
            x += self.type_ids_embedding(type_ids) 

        if self.dpe_ids_embedding:
            x += self.dpe_ids_embedding(dpe_ids)
        
        x = x.view(B*S, -1, self.args.embed_dim) 
            
        if self.pos_encoder:   
            x = self.pos_encoder(x) # (B, S, W, E) -> (B*S, W, E)
        x = self.layer_norm(x)
        return x


class TransformerEventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_dim = args.pred_dim

        encoder_layers = TransformerEncoderLayer(
            args.embed_dim,
            args.n_heads,   
            args.embed_dim*4,
            args.dropout,
            batch_first=True
        )

        self.transformer_encoder = TransformerEncoder( 
            encoder_layers, args.n_layers
        )

        self.post_encode_proj = nn.Linear(args.embed_dim, self.pred_dim)

    def forward(self, all_codes_embs, input_ids, **kwargs):

        B, S, _ = input_ids.size()
        src_pad_mask = (
            input_ids.view(B * S, -1).eq(0).to(all_codes_embs.device)
        )  # (B, S, W) -> (B*S, W)
        encoder_output = self.transformer_encoder(
            all_codes_embs, src_key_padding_mask=src_pad_mask
        )
        encoder_output[src_pad_mask] = 0
        encoder_output = torch.div(encoder_output.sum(dim=1), (encoder_output!=0).sum(dim=1))
        net_output = self.post_encode_proj(encoder_output).view(
            B, -1, self.pred_dim
        )

        return net_output


class Aggregator(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.slopes = None
        if self.args.time_embedding == 'alibi_time_sym':
            self.slopes = torch.Tensor(get_slopes(args.n_heads, args.alibi_const))
        else:
            self.pos_encoder = PositionalEncoding(
                args.pred_dim, args.dropout, args.max_seq_len
            )
            self.layer_norm = nn.LayerNorm(args.embed_dim, eps=1e-12)

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,
            args.pred_dim * 4,
            args.dropout,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, args.n_layers
        )

    def forward(self, events, input_ids, times, **kwargs):
        # input_ids: (B, S) (B x S, W ) -> (Bx s, W) -> (B, s, W)

        B, S = input_ids.shape[0], input_ids.shape[1]
        src_pad_mask = input_ids[:, :, 1].eq(0).to(events.device)

        src_mask = None

        if self.slopes is not None:
            src_mask = -(times.cpu().unsqueeze(1).repeat(1, S, 1) - times.cpu().unsqueeze(2)).abs()
            src_mask = torch.einsum("i, jkl -> jikl", self.slopes, src_mask).reshape(-1, S, S).to(events.device)

        elif self.pos_encoder is not None:
            events = self.layer_norm(self.pos_encoder(events))
        encoder_output = self.transformer_encoder(
            events, mask=src_mask, src_key_padding_mask=src_pad_mask
        )

        return encoder_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
def get_slopes(n, alibi_const):
    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - alibi_const)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            _get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2, alibi_const)[0::2][
                : n - closest_power_of_2
            ]
        )


class PredOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.final_proj = nn.ModuleDict()

        for task in self.args.pred_tasks:
            self.final_proj[task.name] = nn.Linear(
                args.pred_dim, task.num_classes
            )

    def forward(self, x, input_ids, **kwargs): #n_iter
        B, S = input_ids.size(0), input_ids.size(1)
        if self.args.pred_pooling == "cls":
            x = x[:, 0, :]
        elif self.args.pred_pooling == "mean":
            mask = ~input_ids[:, :, 1].eq(0)

            mask = mask.unsqueeze(dim=2).to(x.device).expand(B, S, self.args.pred_dim)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)

            extract_latent = kwargs.get('extract_latent')
            data_type = kwargs.get('data_type')
            subject = kwargs.get('subject')
            if extract_latent and data_type == 'test':
                n_iter = kwargs.get('n_iter')
                latent_save_dir = kwargs.get("latent_save_dir")
                
                torch.save(x, os.path.join(latent_save_dir, f'{subject}_{n_iter}.pt'))
        
        preds = dict()

        for k, layer in self.final_proj.items():
            preds[k] = layer(x)

        return {"pred_output": preds}
