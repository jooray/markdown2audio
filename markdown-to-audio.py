import re
import argparse
import numpy as np
import soundfile as sf
import markdown
from bs4 import BeautifulSoup
from f5_tts_mlx.generate import generate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pywhispercpp.model import Model  # Importing pywhispercpp
from difflib import SequenceMatcher
import os
import tempfile
import time
from tqdm import tqdm  # Import tqdm for progress bar

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Function to remove markdown links
def remove_links(markdown_text):
    # Replace [text](link) with text
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)

# Function to read and parse markdown text
def parse_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Remove links
    markdown_text = remove_links(markdown_text)

    # Convert markdown to HTML
    html = markdown.markdown(markdown_text)

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Extract headings and paragraphs
    texts = []
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        text = element.get_text(strip=True)
        if text:
            texts.append(text)
    return texts

# Function to split text into sentences using NLTK
def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to split long sentences into smaller chunks semantically
def split_long_sentences(sentences, max_length):
    result = []
    for sentence in sentences:
        if len(sentence) <= max_length:
            result.append(sentence)
        else:
            # Try splitting at punctuation marks for semantic splitting
            split_points = [',', ';', ':']
            chunks = [sentence]
            for punct in split_points:
                temp_chunks = []
                for chunk in chunks:
                    if len(chunk) > max_length:
                        sub_chunks = [s.strip() for s in chunk.split(punct)]
                        temp_chunks.extend(sub_chunks)
                    else:
                        temp_chunks.append(chunk)
                chunks = temp_chunks

            # If still too long, split at word boundaries
            final_chunks = []
            for chunk in chunks:
                if len(chunk) <= max_length:
                    final_chunks.append(chunk)
                else:
                    words = word_tokenize(chunk)
                    current_chunk = ''
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= max_length:
                            if current_chunk:
                                current_chunk += ' ' + word
                            else:
                                current_chunk = word
                        else:
                            final_chunks.append(current_chunk)
                            current_chunk = word
                    if current_chunk:
                        final_chunks.append(current_chunk)
            result.extend(final_chunks)
    return result

# Function to calculate similarity ratio
def similarity_ratio(a, b):
    matcher = SequenceMatcher(None, a, b)
    return matcher.ratio()

# Function to generate audio and find the best match
def generate_best_audio(text, args, stt_model):
    print(f"Processing text: {text}")
    if not text.endswith('.'):
        text += '.'

    max_duration_seconds = 30.0  # Maximum allowed duration in seconds

    best_similarity = 0.0
    best_audio_path = None
    duration_seconds = None  # Initialize to None
    retries = 0
    max_retries = 5  # To prevent infinite loops

    temp_files = []  # Keep track of temporary files to delete later

    while retries < max_retries:
        # Create a unique temporary file for each attempt
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # We will write to this file later
        temp_files.append(temp_path)

        generate_args = {
            'generation_text': text,
            'model_name': 'lucasnewman/f5-tts-mlx',
            'output_path': temp_path,
            'seed': args.seed,
            # Do not set 'duration' on first attempt
        }
        if duration_seconds is not None:
            # For subsequent attempts, set duration
            generate_args['duration'] = duration_seconds

        if args.ref_audio and args.ref_text:
            generate_args['ref_audio_path'] = args.ref_audio
            generate_args['ref_audio_text'] = args.ref_text
        elif args.ref_audio or args.ref_text:
            raise ValueError("Both reference audio and reference text must be provided together.")
        # Else, do not pass ref_audio_path and ref_audio_text

        # Generate audio
        generate(**generate_args)

        # Read the generated audio to get its duration
        audio_data, _ = sf.read(temp_path)
        generated_duration_seconds = len(audio_data) / 24000  # SAMPLE_RATE is 24000 Hz

        # Transcribe the audio
        segments = stt_model.transcribe(temp_path)
        transcribed_text = ''.join([segment.text for segment in segments]).strip()

        # Output the transcription for debugging
        print(f"Transcribed text: {transcribed_text}")

        # Clean up transcribed text and original text for comparison
        clean_transcribed_text = re.sub(r'\s+', ' ', transcribed_text.lower())
        clean_original_text = re.sub(r'\s+', ' ', text.lower())

        # Compare the ends of the texts
        N = min(50, len(clean_original_text))  # Adjust N as needed
        transcribed_end = clean_transcribed_text[-N:]
        original_end = clean_original_text[-N:]

        # Compute similarity
        similarity = similarity_ratio(transcribed_end, original_end)

        print(f"Attempt {retries + 1}: Similarity (last {N} chars): {similarity:.2f}")

        # Check if similarity has improved
        if similarity > best_similarity:
            best_similarity = similarity
            best_audio_path = temp_path

            # If similarity is high enough, accept the audio
            if similarity >= 0.9:
                print("Audio accepted.")
                break
        else:
            # If similarity hasn't improved, use previous best audio
            print("No improvement in similarity.")
            break

        # For next attempt, increase duration by 20%, respecting max_duration_seconds
        if duration_seconds is None:
            duration_seconds = min(generated_duration_seconds * 1.2, max_duration_seconds)
        else:
            duration_seconds = min(duration_seconds * 1.2, max_duration_seconds)

        if duration_seconds > max_duration_seconds:
            print("Reached maximum duration limit.")
            break

        retries += 1
        print(f"Increasing duration to {duration_seconds:.2f} seconds and retrying...")

    if best_audio_path is not None:
        audio, _ = sf.read(best_audio_path)
    else:
        # If no good audio was generated, use the last attempt
        print(f"Warning: Could not generate complete audio for text: {text}")
        # Use the last generated audio
        audio, _ = sf.read(temp_path)

    # Clean up temporary files except the best one
    for file_path in temp_files:
        if os.path.exists(file_path) and file_path != best_audio_path:
            os.remove(file_path)

    return audio

# Function to generate audio from text with verification
def text_to_audio(texts, args):
    audio_segments = []
    # Initialize the speech-to-text model
    stt_model = Model('base.en', n_threads=4)  # Adjust n_threads as needed

    total_chunks = len(texts)
    start_time = time.time()

    # Use tqdm for progress bar
    with tqdm(total=total_chunks, desc='Generating audio', unit='chunk') as pbar:
        for idx, text in enumerate(texts):
            audio = generate_best_audio(text, args, stt_model)
            audio_segments.append(audio)
            # Add pause
            if args.pause > 0:
                pause = np.zeros(int(args.pause * 24000))  # SAMPLE_RATE is 24000 Hz
                audio_segments.append(pause)

            # Update progress bar
            pbar.update(1)

            # Calculate elapsed time and estimated remaining time
            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / (idx + 1)
            remaining_chunks = total_chunks - (idx + 1)
            remaining_time = avg_time_per_chunk * remaining_chunks

            # Update progress bar postfix with time estimations
            pbar.set_postfix({
                'Elapsed': f"{elapsed_time:.2f}s",
                'Remaining': f"{remaining_time:.2f}s"
            })

    return np.concatenate(audio_segments)

# Function to save the final audio
def save_audio(audio, file_path):
    sf.write(file_path, audio, 24000)  # SAMPLE_RATE is 24000 Hz

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Generate spoken audio from a markdown file.")
parser.add_argument("input", help="Input markdown file path.")
parser.add_argument("output", help="Output audio file path.")
parser.add_argument("--ref-audio", default="", help="Reference audio file path.")
parser.add_argument("--ref-text", default="", help="Text spoken in the reference audio.")
parser.add_argument("--pause", type=float, default=0.5, help="Pause duration in seconds between sentences.")
parser.add_argument("--max-length", type=int, default=500, help="Maximum length of text chunk.")
parser.add_argument("--seed", type=int, default=None, help="Seed for noise generation.")
args = parser.parse_args()

# Main execution
if __name__ == "__main__":
    texts = parse_markdown(args.input)
    # Split texts into sentences using NLTK
    sentences = []
    for text in texts:
        sents = split_into_sentences(text)
        sentences.extend(sents)
    # Split long sentences semantically
    max_length = args.max_length
    sentences = split_long_sentences(sentences, max_length)
    concatenated_audio = text_to_audio(sentences, args)
    save_audio(concatenated_audio, args.output)
