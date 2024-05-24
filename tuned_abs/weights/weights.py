import os
import gdown

_WEIGHTS_DIR = os.path.dirname(__file__)

NAME_TO_WEIGHTS = {
    'finetuned': os.path.join(_WEIGHTS_DIR, 'finetuned.pt'),
    'finetunedvalid': os.path.join(_WEIGHTS_DIR, 'finetunedvalid.pt'),
    'singlesequence': os.path.join(_WEIGHTS_DIR, 'singlesequence.pt'),
}

# NAME_TO_WEIGHTS = {
#     'finetuned': '/titan/bohdan/parameters/finetuned.pth',
#     'finetunedvalid': '/titan/bohdan/parameters/finetuned_valid.pth',
#     'singlesequence': '/titan/bohdan/parameters/singlesequence.pth',
# }

NAME_TO_URL = {
    'finetuned': 'https://drive.google.com/uc?id=1J3kJ__OmYUl9jO8V5uKiVS8tEM25ugFB',
    'finetunedvalid': 'https://drive.google.com/uc?id=1phiUxSK9TGXSO3MsCa5u2UwC2F_TY3-a',
    'singlesequence': 'https://drive.google.com/uc?id=1Bf_iEB6jawKEI1lDUOxaCJGA_raVcpDh',
}


def get_weights_path(model_name: str, quiet: bool = False):
    weights_path = NAME_TO_WEIGHTS[model_name.lower()]
    if not os.path.exists(weights_path):
        url = NAME_TO_URL[model_name]
        gdown.download(url, weights_path, quiet=quiet)

    return weights_path

