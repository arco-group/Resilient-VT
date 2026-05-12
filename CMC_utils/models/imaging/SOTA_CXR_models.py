import torch
import torchxrayvision as xrv

from CMC_utils.miscellaneous import do_really_nothing
from CMC_utils.models import Extractor, freeze_params, set_pretrained_attribute, remove_DataParallel_module

__all__ = ["get_CXRdensenet", "get_CXRresnet"]

def get_CXRdensenet(weights="densenet121-res224-all", num_classes=None, freeze_model_params=False, extractor=False, op_threshs=None, apply_sigmoid=False, **kwargs) -> torch.nn.Module:
    """
    Get DenseNet model pretrained on CXR data
    Parameters
    ----------
    weights : str
    num_classes : int
    freeze_model_params : bool
    extractor : bool
    d_token : int
    op_threshs : list
    apply_sigmoid : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    
    model = xrv.models.DenseNet(weights=weights, op_threshs=op_threshs, apply_sigmoid=apply_sigmoid)
    setattr(model, "input_size", (1, 224, 224))
    setattr(model, "d_token", 1024)

    set_pretrained_attribute(model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes and not extractor:
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model.op_threshs = None if len(model.op_threshs) == 18 else model.op_threshs
        setattr(model, "output_size", num_classes)
    elif extractor:
        num_ftrs = model.classifier.in_features
        setattr(model, "output_size", (num_ftrs,))
        model.classifier = torch.nn.Identity()
        model.op_threshs = None if len(model.op_threshs) == 18 else model.op_threshs
        model = Extractor(model, num_ftrs, **kwargs)
        return model
    else:
        setattr(model, "output_size", model.classifier.out_features)

    return model


def get_CXRresnet(weights: str = "resnet50-res512-all", num_classes: int = None, load_weights: bool = False, freeze_model_params: bool = False, extractor: bool = False, checkpoint_path: str = None, extract_model: bool = False, apply_sigmoid: bool = False, **kwargs) -> torch.nn.Module:
    """
    Get DenseNet model pretrained on CXR data
    Parameters
    ----------
    weights : str
    num_classes : int
    load_weights : bool
    freeze_model_params : bool
    extractor : bool
    checkpoint_path : str
    extract_model : bool
    apply_sigmoid : bool
    kwargs : dict

    Returns
    -------
    torch.nn.Module
    """
    use_weights = weights is not None
    if not use_weights:
        weights = "resnet50-res512-all"

    model = xrv.models.ResNet(weights=weights, apply_sigmoid=apply_sigmoid)

    load_options = {False: do_really_nothing, True: set_pretrained_attribute}
    load_options[use_weights or load_weights](model)

    freeze_options = {False: do_really_nothing, True: freeze_params}
    freeze_options[freeze_model_params](model)

    if num_classes:
        num_ftrs = model.model.fc.in_features
        model.model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
        model.op_threshs = None

    if extract_model:
        model = model.model

    setattr(model, "input_size", (512, 512))
    setattr(model, "d_token", 2048)
    if num_classes:
        setattr(model, "output_size", num_classes)

    if load_weights and checkpoint_path is not None:
        model_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model_state_dict = remove_DataParallel_module(model_state_dict)

        model.load_state_dict(model_state_dict)

        load_options = {False: do_really_nothing, True: set_pretrained_attribute}
        load_options[load_weights](model)

        freeze_options = {False: do_really_nothing, True: freeze_params}
        freeze_options[freeze_model_params](model)

    if extractor:
        if extract_model:
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Identity()
        else:
            num_ftrs = model.model.fc.in_features
            model.model.fc = torch.nn.Identity()
            model.op_threshs = None
        # setattr(model, "input_size", (1, 512, 512))
        setattr(model, "d_token", num_ftrs)
        setattr(model, "output_size", (num_ftrs,))
        model = Extractor(model, num_ftrs, **kwargs)
        return model
    else:
        if extract_model:
            setattr(model, "output_size", model.fc.out_features)
        else:
            setattr(model, "output_size", model.model.fc.out_features)

    return model


if __name__ == "__main__":
    pass
