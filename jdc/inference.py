import torch


def read_jdc_model(model_path: str):
    """
    Reads the model for evaluation (inference).

    Args:
        model_path(str): model path

    Returns:
        model object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_obj = torch.load(model_path, map_location=device)
    model_obj.eval()
    return model_obj


if __name__ == '__main__':
    model = read_jdc_model('./best/model/e051_JDCNet.pth')
    print(model)

    dummy = torch.randn((1, 1, 31, 513))
    print(model(dummy))
