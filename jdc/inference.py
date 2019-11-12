import torch
import numpy as np
from .util import log_magnitude_spectrogram


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


def spectrogram_chunks(log_mag_spec, chunk_size=31):
    # log_mag_spec: (freq_bin, frames)
    log_mag_spec = log_mag_spec.T  # swap axes: (frames, freq_bin)

    num_frames = log_mag_spec.shape[0]

    # split the spectrogram in chunks
    for start_idx in range(0, num_frames, chunk_size):
        end_idx = start_idx + chunk_size
        chunk = log_mag_spec[start_idx:end_idx]

        # most probably the chunk is short on frames at the end of audio
        chunk_length = chunk.shape[0]
        if chunk_length != chunk_size:
            raise StopIteration

        # save the processed data to pickle file
        # swap axes: (freq_bin, chunk_size) => (chunk_size, freq_bin)
        yield chunk


def extract_melody(audio_file_path: str, model):
    # in = audio_file_path
    log_mag = log_magnitude_spectrogram(audio_file_path)

    melody_line = []
    for spec_chunk in spectrogram_chunks(log_mag):
        # convert ndarray to torch tensor as model input
        # (31, 513) => (1, 1, 31, 513)
        in_tensor = torch.from_numpy(spec_chunk[np.newaxis, np.newaxis, :])
        print(in_tensor.size())
        out_classification, out_detection = model(in_tensor)

        _, pitch_classes = out_classification.max(dim=2)
        _, isvoice = out_detection.max(dim=2)
        isvoice = isvoice == 1
        print(pitch_classes, isvoice)

    # out = midi sequence of melody lines
    return melody_line


if __name__ == '__main__':
    model = read_jdc_model('./best/model/e051_JDCNet.pth')
    print(model)

    # dummy = torch.randn((1, 1, 31, 513))
    # print(model(dummy))

    extract_melody('FacesOnFilm_WaitingForGa_MIX.wav', model)
