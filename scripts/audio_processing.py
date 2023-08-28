import os
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence


def split_audio_into_chunks(audio_file_path, output_dir="chunks",
                            min_silence_len=1000, silence_thresh=-40):
    """
    Splits an audio file into chunks based on silence intervals.

    Parameters:
    - audio_file_path: Path to the main audio file.
    - output_dir: Directory where chunk files will be saved.
    - min_silence_len: Minimum length of silence to be considered a split point (in ms).
    - silence_thresh: Silence threshold in dB.

    Returns:
    - List of paths to the saved audio chunks.
    """

    # Load the audio file
    audio = AudioSegment.from_wav(audio_file_path)

    # Split audio based on silence
    chunks = split_on_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(output_dir, f"chunk_{i}.wav")
        chunk.export(chunk_file_path, format="wav")
        chunk_files.append(chunk_file_path)

    return chunk_files


def transcribe_audio_chunks_in_parallel(audio_chunk_paths, model_name="base.en", batch_size=16):
    """
    Transcribes a list of audio chunks in parallel using the Whisper model.

    Parameters:
    - audio_chunk_paths: List of paths to the audio chunk files.
    - model_name: Name of the Whisper model to use. Default is "base.en".
    - batch_size: Number of audio chunks to process in a single batch.

    Returns:
    - List of transcribed texts for each audio chunk.
    """

    # Load the Whisper model and set to GPU
    model = whisper.load_model(model_name).to("cuda")

    # Convert audio chunks to log Mel spectrograms or any other preprocessing required
    # ... (This step will be based on the LibriSpeech notebook details)

    # Create a DataLoader for batch processing
    loader = DataLoader(audio_chunk_paths, batch_size=batch_size)

    # List to store transcriptions
    transcriptions = []

    # Process audio chunks in batches
    for batch in tqdm(loader):
        # Transcribe the batch using Whisper
        results = model.decode(batch)
        transcriptions.extend([result.text for result in results])

    return transcriptions
