from functools import reduce

import torch
from hydra.utils import instantiate

from CMC_utils.miscellaneous import recursive_cfg_substitute

import logging
log = logging.getLogger(__name__)

__all__ = ["MultimodalLearner"]


class MultimodalLearner(torch.nn.Module):
    """
    MultimodalLearner is a generic class for multimodal learning. It takes as input a list of models and outputs a
    single prediction. The models can be of any type, as long as they have a forward method that takes a single input
    and returns a single output. The output of the models is concatenated and fed to a fully connected layer.
    """
    def __init__(self, ms_models, shared_net, fusion_mode: str = "concat", **_):
        super(MultimodalLearner, self).__init__()
        self.fusion_mode = fusion_mode
        for model_id in ms_models.keys():
            if ms_models[model_id]["name"].startswith("TabNet"):
                param_dict = dict(output_dim=ms_models[model_id]["init_params"].get("input_dim", None))
            elif ms_models[model_id]["name"].startswith(("TabTransformer", "FTTransformer")):
                param_dict = dict( dim_out=ms_models[model_id]["init_params"].get("num_continuous", 0) + len(ms_models[model_id]["init_params"].get("cat_idxs", [])), extractor=True )
            else:
                param_dict = dict(extractor=True)

            ms_models[model_id] = recursive_cfg_substitute(ms_models[model_id], param_dict)

        self.ms_models = torch.nn.ModuleList()
        for model_id, model in ms_models.items():
            self.ms_models.append(instantiate(model["init_params"], _recursive_=False))

        self.input_size = [model.input_size for model in self.ms_models]

        self.ms_output_sizes = [model.output_size for model in self.ms_models]

        if shared_net["name"].startswith("TabNet"):
            param_dict = dict( cat_idxs = [], cat_dims = [], input_dim= sum(self.input_size) )
        elif shared_net["name"].startswith(("TabTransformer", "FTTransformer")):
            param_dict = dict( cat_idxs = [], categories = [], num_continuous = sum(self.ms_output_sizes), embed_input= False )
        elif shared_net["name"].startswith("naim"):
            d_token = self.ms_models[0].d_token
            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(torch.tensor([s[0] for s in self.ms_output_sizes]))//d_token)
        elif shared_net["name"].startswith("maria"):
            d_token = self.ms_models[0].d_token
            tokens_per_modality = torch.tensor([s[0] for s in self.ms_output_sizes], dtype=torch.int64)//d_token
            param_dict = dict(embed_input= False, d_token=d_token, input_size=torch.sum(tokens_per_modality), ntokens_per_modality=tokens_per_modality)
            # for i, (inp_size, out_size) in enumerate(zip(self.input_size, self.ms_output_sizes)):
                #if isinstance(inp_size, list) or isinstance(inp_size, tuple):
                #    if isinstance(out_size, list) or isinstance(out_size, tuple):
                #        param_dict["input_size"][i] = out_size[0]
                #    else:
                #        param_dict["input_size"][i] = out_size
                # param_dict["input_size"][i] = torch.sum(out_size[0])
            # ms_output_size = int(sum(self.ms_output_sizes) / d_token)
        elif shared_net["name"].startswith("MLP"):
            if self.fusion_mode in ("sum", "max"):
                param_dict = {"input_size": self.ms_models[0].d_token}
            else:
                param_dict = {"input_size": sum(reduce(lambda x, y: x + y if y else x, self.ms_output_sizes))}
        else:
            param_dict = dict()

        shared_net = recursive_cfg_substitute(shared_net, param_dict)
        self.shared_net = instantiate(shared_net["init_params"], _recursive_=False)

        self.output_size = self.shared_net.output_size

    def forward(self, *multiple_inputs):
        hidden_representations = list()
        for inputs, model in zip(multiple_inputs, self.ms_models):
            hidden_representations.append(model(inputs))

        B = hidden_representations[0].shape[0]
        hidden_representations = torch.cat([hidden.view(B, -1) for hidden in hidden_representations], dim=1)

        if self.fusion_mode == "sum":
            hidden_representations = hidden_representations.view(B, -1, self.ms_models[0].d_token)
            hidden_representations = torch.sum(hidden_representations, dim=1)
        elif self.fusion_mode == "max":
            hidden_representations = hidden_representations.view(B, -1, self.ms_models[0].d_token)
            hidden_representations, _ = torch.max(hidden_representations, dim=1)

        out = self.shared_net(hidden_representations, multiple_inputs)

        return out.squeeze()


if __name__ == "__main__":
    pass
