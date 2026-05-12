import math
import torch
from typing import List
# from skimage.filters.rank import modal
from torch import Tensor
from torch.nn import Sigmoid
import torch.nn.functional as F
from .tabular_tokenizer import CategoricalFeatureTokenizer
from typing import Tuple, Optional

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import Extractor, set_pretrained_attribute, freeze_params, remove_DataParallel_module

__all__ = ["NAIM", "MARIA", "get_naim_pretrained"]


class TabularMasker:
    def __init__(self, missing_value: str = "-inf"):
        missing_value_options = {"-inf": -torch.inf, "~inf": -1e9}
        self.missing_value = missing_value_options[missing_value]

    def _tabular_sample_mask(self, sample: Tensor):
        mask = torch.clone(sample)
        mask[~torch.isnan(sample)] = 0
        mask[torch.isnan(sample)] = 1
        return mask

    def mask(self, data: Tensor, **_):
        masks = Tensor().to(data.device)

        for sample in data:
            sample_mask = self._tabular_sample_mask(sample).to(torch.bool)

            sample_mask = sample_mask.repeat(sample_mask.shape[0], 1)

            masks = torch.cat([masks, sample_mask.unsqueeze(dim=0)], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        return masks  # , masks.transpose(-2, -1)

    def modality_mask(self, embeddings: Tensor, data: Tensor, use_cls_token: bool = False, **_):
        masks = Tensor().to(embeddings.device)
        n_modalities = len(data) + int(use_cls_token)
        for sample_idx, embedding in enumerate(embeddings):

            if use_cls_token:
                sample_mask = torch.zeros(1, 1, n_modalities).to(embeddings.device)
            else:
                sample_mask = torch.Tensor().to(embeddings.device)

            for modality_data in data:
                sample = modality_data[sample_idx]
                if torch.isnan(sample).all():
                    ith_modality_mask = torch.ones(1, 1, n_modalities).to(embeddings.device)
                else:
                    ith_modality_mask = torch.zeros(1, 1, n_modalities).to(embeddings.device)
                sample_mask = torch.cat([sample_mask, ith_modality_mask], dim=1)
            masks = torch.cat([masks, sample_mask], dim=0)

        masks = torch.masked_fill(masks, masks.to(torch.bool), self.missing_value)

        return masks, masks.transpose(-2, -1)


class Attention(torch.nn.Module):

    def __init__(self, dropout_rate: float = 0.0):
        super(Attention, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, q: Tensor, k: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:

        B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        attn = torch.add(attn, attn_mask.transpose(-2, -1))
        attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)

        return attn


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-Head Attention module.
    """
    def __init__(self,
                 input_size: int,
                 num_heads: int,
                 bias: bool = True,
                 activation: str = "relu",
                 dropout_rate: float = 0.0):

        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, f"`input_size`({input_size}) should be divisible by `num_heads`({num_heads})"

        self.input_size = input_size
        self.num_heads = num_heads
        self.bias = bias
        activation_options = dict(relu=F.relu, gelu=F.gelu)
        self.activation = activation_options[activation]
        self.dropout_rate = dropout_rate

        self.linear_q = torch.nn.Linear(input_size, input_size, bias)
        self.linear_k = torch.nn.Linear(input_size, input_size, bias)
        self.linear_v = torch.nn.Linear(input_size, input_size, bias)
        self.linear_o = torch.nn.Linear(input_size, input_size, bias)
        self.attn = Attention(dropout_rate)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = torch.repeat_interleave(mask, self.num_heads, 0 )

        y, attn_scores = self._scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def _reshape_to_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x: Tensor) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def _scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        attn = self.attn(q, k, attn_mask)
        """B, Nt, E = q.shape
        q = q / math.sqrt(E)

        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))

        attn = F.softmax(attn, dim=-1)

        if attn_mask_2 is not None:
            attn = torch.add(attn, attn_mask_2)
            attn = F.relu(attn)

        if self.dropout_rate > 0.0:
            attn = F.dropout(attn, p=self.dropout_rate)"""

        output = torch.bmm(attn, v)

        return output, attn


class EncoderBlock(torch.nn.Module):
    """
    Encoder block of the Transformer.
    """
    def __init__(self, emb_dim, ff_dim, num_heads, bias: bool = False, activation: str = "relu", dropout_rate: float = 0.0):
        super(EncoderBlock, self).__init__()

        self.layer_norm_1 = torch.nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads, bias=bias, activation=activation, dropout_rate=dropout_rate )
        self.layer_norm_2 = torch.nn.LayerNorm(emb_dim)

        activation_options = dict( relu= torch.nn.ReLU, gelu= torch.nn.GELU )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ff_dim),
            activation_options[activation](),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, emb_dim),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor, mask: Tensor = None):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x, mask=mask)
        x = self.layer_norm_2(x)
        x = x + self.ff(x)
        return x


class NAIM(torch.nn.Module):
    """
    NAIM model for tabular data.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 cat_idxs: list,
                 cat_dims: list,
                 d_token: int,
                 embedder_initialization: str,
                 bias: bool,
                 missing_value: str = "-inf",
                 num_heads: int = 12,
                 feedforward_dim: int = 1000,
                 dropout_rate: float = 0.1,
                 activation: str = "relu",
                 num_layers: int = 12,
                 # embed_input: bool = True,
                 embed_vector_fun: str = "cat",
                 limit_regression_output: bool = False,
                 extractor: bool = False):

        super(NAIM, self).__init__()

        self.input_size = input_size
        self.cat_idxs = cat_idxs if cat_idxs else [-1]
        self.cat_dims = cat_dims if cat_dims else [-1]
        self.d_token = d_token
        self.embedder_initialization = embedder_initialization
        self.bias = bias
        self.missing_value = missing_value
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_layers = num_layers
        # self.embed_input = embed_input
        self.embed_vector_fun = embed_vector_fun  # "cat", "mean", "clstoken"
        self.limit_regression_output = limit_regression_output
        self.extractor = extractor

        self.cls_token = None
        if self.embed_vector_fun == "clstoken":
            self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.d_token))

        if self.extractor:
            if self.embed_vector_fun in ("clstoken", "mean"):
                self.output_size = (self.d_token,)
            else:
                self.output_size = (input_size * d_token, )
        else:
            if self.embed_vector_fun in ("clstoken", "mean"):
                self.embed_size = self.d_token
            else:
                self.embed_size = input_size * d_token
            self.output_size = (output_size, )

        # EMBEDDERS initializations
        #if self.embed_input:
        j = 0
        self.embeddings = torch.nn.ModuleList()
        common_params = dict( d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization )
        for i in range(input_size):
            is_categorical_feature = i in self.cat_idxs
            feature_type_params = { True: dict(cardinalities = [self.cat_dims[j] + 1], padding_idx=self.cat_dims[j]),
                                    False: dict(cardinalities = [2], padding_idx=1) }

            j = j + (is_categorical_feature * (i != self.cat_idxs[-1]))
            embedding = CategoricalFeatureTokenizer( **common_params, **feature_type_params[ is_categorical_feature ])

            self.embeddings.append(embedding)
        #else:
        #    self.embeddings = torch.nn.Identity()

        # MASKER initialization
        self.attention_mask = TabularMasker(self.missing_value)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

        self.encoder = torch.nn.ModuleList([EncoderBlock(self.d_token, self.feedforward_dim, self.num_heads, bias=self.bias, activation=self.activation, dropout_rate=self.dropout_rate) for _ in range(self.num_layers)])

        self.norm = torch.nn.LayerNorm(self.d_token)

        # classifier
        if not self.extractor:
            if self.output_size[0] > 1:
                self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))
            else:
                if limit_regression_output:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]), Sigmoid())
                else:
                    self.classifier = torch.nn.Sequential(torch.nn.Linear(self.embed_size, self.output_size[0]))

    def feature_embedding(self, x: Tensor):
        j = 0
        embeddings = Tensor().to(x.device)
        for feature_idx in list(range(x.shape[1])):
            if feature_idx in self.cat_idxs:
                single_feature = torch.nan_to_num(x[:, feature_idx], nan=self.cat_dims[j]).to(torch.int64)
                feature_values = None
                j += 1
            else:
                single_feature = torch.isnan(x[:, feature_idx]).to(torch.int64)
                feature_values = torch.nan_to_num(x[:, feature_idx], nan=0)
            single_feature_embedding = self.embeddings[feature_idx](single_feature, feature_values)

            single_feature_embedding = torch.swapaxes(single_feature_embedding, 0, 1)
            embeddings = torch.cat([embeddings, single_feature_embedding], dim=1)
        return embeddings

    def forward(self, x, _=None):
        embeddings = self.feature_embedding(x)
        if self.embed_vector_fun == "clstoken":
            cls_tokens = self.cls_token.repeat(embeddings.shape[0], 1, 1).to(embeddings.device)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

            x = torch.cat([torch.ones(x.shape[0], 1).to(x.device), x], dim=1)

        masks = self.attention_mask.mask(x)

        # transformer
        for encoder_layer in self.encoder:
            embeddings = encoder_layer(embeddings, mask=masks)

        embeddings = self.norm(embeddings)

        if self.embed_vector_fun == "mean":
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            embeddings = embeddings.nanmean(dim=1)
            embeddings = torch.nan_to_num(embeddings, nan=0)
        elif self.embed_vector_fun == "clstoken":
            embeddings = embeddings[:, 0]
        else:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        if self.extractor:
            return embeddings

        # classifier
        logits = self.classifier(embeddings)

        return logits


class MARIA(NAIM):
    def __init__(self, *args,  n_modalities: int, ntokens_per_modality: List[int], mask_missing_values: bool = True, **kwargs):  # ntokens_per_modality: List[int],
        super(MARIA, self).__init__(*args, **kwargs)
        self.n_modalities = n_modalities
        self.ntokens_per_modality = ntokens_per_modality
        self.mask_missing_values = mask_missing_values

        self.masking_options = {False: self.attention_mask.modality_mask, True: self.attention_mask.mask}
        # self.modality_tokens = torch.nn.Embedding(self.n_modalities + 1, self.d_token, padding_idx=-1)
        self.modality_tokens = CategoricalFeatureTokenizer( cardinalities=[self.n_modalities+1], d_token=self.d_token, bias=self.bias, initialization=self.embedder_initialization, padding_idx=-1 )

    def forward(self, x, original_x=None):
        batch_size = x.shape[0]
        embeddings = x.view(batch_size, -1, self.d_token)
        modalities_matrix = []
        for i in range(self.n_modalities):
            modalities_matrix.append(self.modality_tokens(torch.tensor([i]*self.ntokens_per_modality[i], dtype=torch.int64).to(embeddings.device)))
        modalities_matrix = torch.cat(modalities_matrix, dim=1).repeat(batch_size, 1, 1)
        embeddings = modalities_matrix * embeddings

        # elif self.embed_vector_fun == "cat":
        original_tensor = []
        for j, d in enumerate(original_x):
            if len(d.shape) != 2:
                tnsr = torch.empty((batch_size, self.ntokens_per_modality[j]))
                for i, s in enumerate(d):
                    if torch.isnan(s).all():
                        tnsr[i, :] = torch.nan
                    else:
                        tnsr[i, :] = 0
                original_tensor.append(tnsr.to(d.device))
            else:
                if d.shape[1] > self.ntokens_per_modality[j]:
                    d = d[:, :self.ntokens_per_modality[j]]
                original_tensor.append(d)
        # original_x = torch.cat([d if len(d.shape) == 2 else torch.zeros((batch_size, 1)) for d in original_x], dim=1)
        original_tensor = torch.cat(original_tensor, dim=1)

        # use_cls_token = self.embed_vector_fun == "clstoken"
        if self.embed_vector_fun == "clstoken":
            cls_tokens = self.cls_token.repeat(embeddings.shape[0], 1, 1).to(embeddings.device)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            original_tensor = torch.cat([torch.ones(x.shape[0], 1).to(x.device), original_tensor], dim=1)

        if not self.mask_missing_values:
            original_tensor = torch.nan_to_num(original_tensor, nan=0)

        # masks = self.masking_options[use_cls_token](embeddings=embeddings, data=original_x, use_cls_token=use_cls_token)
        masks = self.attention_mask.mask(data=original_tensor)

        # transformer
        for encoder_layer in self.encoder:
            embeddings = encoder_layer(embeddings, mask=masks)

        embeddings = self.norm(embeddings)

        if self.embed_vector_fun == "mean":
            emb_masks = masks[:, 0] != 0
            embeddings[emb_masks] = torch.nan
            embeddings = embeddings.nanmean(dim=1)
            embeddings = torch.nan_to_num(embeddings, nan=0)
        elif self.embed_vector_fun == "clstoken":
            embeddings = embeddings[:, 0]
        else:
            embeddings = embeddings.view(embeddings.shape[0], -1)

        if self.extractor:
            return embeddings

        # classifier
        logits = self.classifier(embeddings)

        return logits


def get_naim_pretrained(checkpoint_path, output_size, load_weights: bool = True, freeze_model_params: bool = False, extractor: bool = False, **kwargs):
    model = NAIM(output_size=output_size, extractor=False, **kwargs)
    model_state_dict = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = remove_DataParallel_module(model_state_dict)
    model.load_state_dict(model_state_dict)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if output_size and not extractor:
        num_ftrs = model.classifier[0].in_features
        model.classifier[0] = torch.nn.Linear(in_features=num_ftrs, out_features=output_size)

    if extractor:
        num_ftrs = model.classifier[0].in_features
        model.classifier = torch.nn.Identity()
        model = Extractor(model, num_ftrs, **kwargs)
        return model

    return model


if __name__ == "__main__":
    pass
