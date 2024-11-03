import re
import argparse
import numpy as np
import soundfile as sf
from f5_tts_mlx.generate import generate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pywhispercpp.model import Model
from difflib import SequenceMatcher
import os
import tempfile
import time
from tqdm import tqdm
from colorama import init, Fore, Style

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

# Initialize colorama
init(autoreset=True)

# Function to remove markdown links
def remove_links(markdown_text):
    # Replace [text](link) with text
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)

def parse_markdown_with_pauses(file_path, heading_pauses, max_length):
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()

    # Remove links
    markdown_text = remove_links(markdown_text)

    # Pause pattern with capturing groups
    pause_pattern = re.compile(r'(<!--\s*pause:(\d+\.?\d*)(ms|s)\s*-->)')

    # Split the markdown text into lines
    lines = markdown_text.split('\n')

    segments = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for heading
        heading_match = re.match(r'^(#{1,6})\s*(.*)', line)
        if heading_match:
            hashes, heading_text = heading_match.groups()
            level = min(len(hashes), 3)
            pause_before = heading_pauses.get(f'h{level}_before', 0)
            pause_after = heading_pauses.get(f'h{level}_after', 0)
            is_heading = True
            heading_level = level

            # Before processing, add pause_before if any
            if pause_before > 0:
                segments.append({'text': '', 'pause_after': pause_before, 'is_heading': False})

            # Process the heading text
            line_segments = process_line_with_pauses(heading_text, pause_pattern, max_length, is_heading, heading_level)
            segments.extend(line_segments)

            # After processing, add pause_after if any
            if pause_after > 0:
                segments.append({'text': '', 'pause_after': pause_after, 'is_heading': False})

        else:
            # Non-heading line
            line_segments = process_line_with_pauses(line, pause_pattern, max_length)
            segments.extend(line_segments)

    return segments

def process_line_with_pauses(line, pause_pattern, max_length, is_heading=False, heading_level=None):
    segments = []
    pos = 0
    for match in pause_pattern.finditer(line):
        start, end = match.span()
        # Text before the pause tag
        if start > pos:
            text = line[pos:start].strip()
            if text:
                sentences = split_into_sentences(text)
                sentences = split_long_sentences(sentences, max_length)
                for sentence in sentences:
                    segment = {
                        'text': sentence.strip(),
                        'pause_before': None,
                        'pause_after': None,
                        'is_heading': is_heading,
                        'heading_level': heading_level
                    }
                    segments.append(segment)
        # Process the pause tag
        duration = match.group(2)
        unit = match.group(3)
        if unit == 'ms':
            pause_duration = float(duration) / 1000.0
        else:
            pause_duration = float(duration)
        segments.append({'text': '', 'pause_after': pause_duration, 'is_heading': False})
        pos = end

    # Text after the last pause tag
    if pos < len(line):
        text = line[pos:].strip()
        if text:
            sentences = split_into_sentences(text)
            sentences = split_long_sentences(sentences, max_length)
            for sentence in sentences:
                segment = {
                    'text': sentence.strip(),
                    'pause_before': None,
                    'pause_after': None,
                    'is_heading': is_heading,
                    'heading_level': heading_level
                }
                segments.append(segment)
    return segments

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
    print(f"{Fore.YELLOW}{Style.BRIGHT}Processing text:{Style.RESET_ALL} {text}")
    if not text.endswith('.'):
        text += '.'

    max_duration_seconds = 30.0  # Maximum allowed duration in seconds

    best_similarity = 0.0
    best_audio_path = None
    duration_seconds = None  # Initialize to None
    retries = 0
    max_retries = 5  # Limit attempts to prevent infinite loops

    temp_files = []  # Track temporary files to delete later

    while retries < max_retries:
        # Create a unique temporary file for each attempt
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        temp_files.append(temp_path)

        generate_args = {
            'generation_text': text,
            'model_name': 'lucasnewman/f5-tts-mlx',
            'output_path': temp_path,
            'seed': args.seed,
            # Do not set 'duration' on first attempt
        }
        if duration_seconds is not None:
            generate_args['duration'] = duration_seconds

        if args.ref_audio and args.ref_text:
            generate_args['ref_audio_path'] = args.ref_audio
            generate_args['ref_audio_text'] = args.ref_text
        elif args.ref_audio or args.ref_text:
            raise ValueError("Both reference audio and reference text must be provided together.")

        # Generate audio
        generate(**generate_args)

        # Read generated audio to get its duration
        audio_data, _ = sf.read(temp_path)
        generated_duration_seconds = len(audio_data) / 24000  # SAMPLE_RATE is 24000 Hz

        # Transcribe the audio
        segments = stt_model.transcribe(temp_path)
        transcribed_text = ''.join([segment.text for segment in segments]).strip()

        # Output transcription for debugging
        print(f"{Fore.YELLOW}{Style.BRIGHT}Transcribed text:{Style.RESET_ALL} {transcribed_text}")

        # Clean up transcribed text and original text for comparison
        clean_transcribed_text = re.sub(r'\s+', ' ', transcribed_text.lower())
        clean_original_text = re.sub(r'\s+', ' ', text.lower())

        # Compute similarity of the whole text
        similarity = similarity_ratio(clean_transcribed_text, clean_original_text)
        print(f"Attempt {retries + 1}: Similarity: {similarity:.2f}")

        # Check if max similarity is achieved
        if similarity == 1.0:
            best_similarity = similarity
            best_audio_path = temp_path
            print("Audio accepted with perfect similarity.")
            break

        # If similarity improves, save this as the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_audio_path = temp_path
        elif similarity < best_similarity:
            # If similarity worsens, stop here and use the previous best
            print("No improvement in similarity; stopping.")
            break

        # For next attempt, increase duration by 20%, respecting max_duration_seconds
        if duration_seconds is None:
            duration_seconds = min(generated_duration_seconds * 1.2, max_duration_seconds)
        else:
            duration_seconds = min(duration_seconds * 1.2, max_duration_seconds)

        if duration_seconds >= max_duration_seconds:
            print("Reached maximum duration limit.")
            break

        retries += 1
        print(f"Increasing duration to {duration_seconds:.2f} seconds and retrying...")

    # Load the best audio from the best similarity attempt
    if best_audio_path is not None:
        audio, _ = sf.read(best_audio_path)
    else:
        # If no good audio was generated, use the last attempt
        print(f"Warning: Could not generate complete audio for text: {text}")
        audio, _ = sf.read(temp_path)

    # Clean up temporary files except the best one
    for file_path in temp_files:
        if os.path.exists(file_path) and file_path != best_audio_path:
            os.remove(file_path)

    return audio

# Function to generate audio from text with verification and write incrementally
def text_to_audio(segments, args):
    # Initialize the speech-to-text model
    stt_model = Model('base.en', n_threads=4)  # Adjust n_threads as needed

    total_chunks = len(segments)
    start_time = time.time()

    sample_rate = 24000  # SAMPLE_RATE is 24000 Hz

    # Open the output audio file in write mode
    with sf.SoundFile(args.output, mode='w', samplerate=sample_rate, channels=1, subtype='PCM_16') as out_file:

        # Use tqdm for progress bar with red color
        with tqdm(total=total_chunks, desc='Generating audio', unit='chunk', colour='red') as pbar:
            prev_pause_after = 0
            for idx, segment in enumerate(segments):
                text = segment['text']
                pause_before = segment.get('pause_before')
                pause_after = segment.get('pause_after')

                # If no specific pause is set, use default
                if pause_before is None:
                    pause_before = args.pause
                if pause_after is None:
                    pause_after = args.pause

                # Calculate inter-segment pause (maximum of pause_after and next pause_before)
                inter_pause = max(prev_pause_after, pause_before)
                if inter_pause > 0:
                    pause_samples = int(inter_pause * sample_rate)
                    out_file.write(np.zeros(pause_samples))

                if text:
                    audio = generate_best_audio(text, args, stt_model)
                    out_file.write(audio)
                else:
                    # If text is empty, it's a pause segment
                    pass  # We've already handled pause_before and pause_after

                # Set prev_pause_after for next iteration
                prev_pause_after = pause_after

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

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Generate spoken audio from a markdown file.")
parser.add_argument("input", help="Input markdown file path.")
parser.add_argument("output", help="Output audio file path.")
parser.add_argument("--ref-audio", default="", help="Reference audio file path.")
parser.add_argument("--ref-text", default="", help="Text spoken in the reference audio.")
parser.add_argument("--pause", type=float, default=0.5, help="Default pause duration in seconds between segments.")
parser.add_argument("--max-length", type=int, default=500, help="Maximum length of text chunk.")
parser.add_argument("--seed", type=int, default=None, help="Seed for noise generation.")

# Add command-line arguments for heading pauses (H1 to H3)
parser.add_argument('--pause-h1-before', type=float, default=2.0,
                    help='Pause before H1 headings in seconds.')
parser.add_argument('--pause-h1-after', type=float, default=0.7,
                    help='Pause after H1 headings in seconds.')
parser.add_argument('--pause-h2-before', type=float, default=1.5,
                    help='Pause before H2 headings in seconds.')
parser.add_argument('--pause-h2-after', type=float, default=0.7,
                    help='Pause after H2 headings in seconds.')
parser.add_argument('--pause-h3-before', type=float, default=0.7,
                    help='Pause before H3 headings in seconds.')
parser.add_argument('--pause-h3-after', type=float, default=0.7,
                    help='Pause after H3 headings in seconds.')

args = parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Collect heading pauses from command-line arguments
    heading_pauses = {
        'h1_before': args.pause_h1_before,
        'h1_after': args.pause_h1_after,
        'h2_before': args.pause_h2_before,
        'h2_after': args.pause_h2_after,
        'h3_before': args.pause_h3_before,
        'h3_after': args.pause_h3_after,
    }

    # Parse the markdown with pauses
    segments = parse_markdown_with_pauses(args.input, heading_pauses, args.max_length)

    # Generate audio and write incrementally
    text_to_audio(segments, args)
