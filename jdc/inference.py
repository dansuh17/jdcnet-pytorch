import torch
import librosa
import pickle
from mido import Message, MidiFile, MidiTrack
import numpy as np
from tqdm import tqdm
from .util import log_magnitude_spectrogram, PitchLabeler


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


def spectrogram_chunks(log_mag_spec: np.ndarray, chunk_size=31):
    """
    Generates partial chunks of spectrogram.
    `chunk_size` provides the frame length of each chunk.

    Args:
        log_mag_spec(np.ndarray): log magnitude spectrogram of size:
            (freq_bins, num_frames)
        chunk_size(int): number of frames the chunk will have

    Yield:
        spectrogram chunk of size: (chunk_size, freq_bins)
    """
    log_mag_spec = log_mag_spec.T  # swap axes: (frames, freq_bin)
    num_frames = log_mag_spec.shape[0]

    # split the spectrogram in chunks
    for start_idx in range(0, num_frames, chunk_size):
        end_idx = start_idx + chunk_size
        chunk = log_mag_spec[start_idx:end_idx]

        # most probably the chunk is short on frames at the end of audio
        chunk_length = chunk.shape[0]
        if chunk_length != chunk_size:
            break

        # save the processed data to pickle file
        # swap axes: (freq_bin, chunk_size) => (chunk_size, freq_bin)
        yield chunk


def extract_melody(audio_file_path: str, model, pitch_labeler: PitchLabeler,
                   save_to: str = None, sr=8000, hop_length=80, n_fft=1024):
    """
    Extract melody from an audio file.

    Args:
        audio_file_path(str): audio file path
        model: JDCNet model
        pitch_labeler(PitchLabeler): pitch labeler
        save_to(str): file name to save the melody info

    Returns:
        melody_line: MIDI number per frame. Value of `-1` represents 'no note'
        frame_times: time values for each frame in seconds (time offset from start)
    """
    # in = audio_file_path
    log_mag = log_magnitude_spectrogram(audio_file_path)
    print(f'Extracting melody from {audio_file_path}')

    melody_line = []
    for spec_chunk in tqdm(spectrogram_chunks(log_mag)):
        # convert ndarray to torch tensor as model input
        # (31, 513) => (1, 1, 31, 513)
        in_tensor = torch.from_numpy(spec_chunk[np.newaxis, np.newaxis, :])
        # (1, 31, num_pitch_classes), (1, 31, 2)
        out_classification, out_detection = model(in_tensor)

        _, pitch_classes = out_classification.max(dim=2)
        pitch_classes = pitch_classes[0]  # first and only sample
        _, isvoice = out_detection.max(dim=2)
        isvoice = isvoice[0] == 1

        num_frames = spec_chunk.shape[0]
        for frame_idx in range(num_frames):
            pitch_label = pitch_classes[frame_idx]
            if not isvoice[frame_idx]:
                pitch_label = pitch_labeler.label_nonvoice

            midi_num = pitch_labeler.midi(pitch_label, round_int=True)
            melody_line.append(midi_num)

    # out = midi sequence, timestamps
    frame_indices = np.arange(len(melody_line))
    frame_times = librosa.core.frames_to_time(
        frame_indices, sr=sr, hop_length=hop_length, n_fft=n_fft)
    print(f'Melody extracted from {audio_file_path}, num_frames: {len(melody_line)}')

    if save_to is not None:
        with open(save_to, 'wb') as f:
            pickle.dump((melody_line, frame_times), f)
        print(f'melody saved to: {save_to}')

    return melody_line, frame_times


def save_midi(melody, timestamps, output_file='mysong.mid'):
    """
    Given per-frame MIDI values and its timestamps, convert to midi file.

    Args:
        melody(list[int]): list of per-frame MIDI numbers.
            Value of `-1` represents 'no note'
        timestamps(list[float]): timestamps for each frames.
            Must have same length as `melody`.
        name(str): midi file name
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=1, time=0))
    tempo = 500000  # microseconds per beat - arbitrary default number (represents 120 BPM)

    # frame = 0.01 sec
    microsecs_per_tick = tempo / mid.ticks_per_beat
    assert(len(melody) == len(timestamps))

    prev_time = 0
    curr_note = -1
    for midi_note, time in zip(melody, timestamps):
        midi_note = int(midi_note)

        if curr_note != midi_note:
            if curr_note != -1:
                track.append(Message('note_off', note=curr_note, velocity=64, time=0))
                print(f'msg=NOTE_OFF, midi={curr_note}, time={time}, tick=0')

            if midi_note != -1:
                delta_time = time - prev_time
                prev_time = time
                ticks = int(delta_time * 1e6 / microsecs_per_tick)
                track.append(Message('note_on', note=midi_note, velocity=64, time=ticks))
                print(f'msg=MIDI_ON, midi={midi_note}, time={time}, tick={ticks}')

        curr_note = midi_note
    mid.save(output_file)
    print(f'MIDI file saved to {output_file}')


if __name__ == '__main__':
    model = read_jdc_model('./best/model/e051_JDCNet.pth')
    print(model)

    # mymelody, timestamps = extract_melody('FacesOnFilm_WaitingForGa_MIX.wav', model, PitchLabeler())
    # mymelody, timestamps = extract_melody('gangnam_style.wav', model, PitchLabeler(), save_to='gangnam_style')
    # save_midi(mymelody, timestamps, output_file='gangnam_style.mid')
