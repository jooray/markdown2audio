import re
import numpy as np
import soundfile as sf
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from pywhispercpp.model import Model
from difflib import SequenceMatcher
import os
import tempfile
import time
from tqdm import tqdm
from colorama import init, Fore, Style
from f5_tts_mlx.generate import generate

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)

# Initialize colorama
init(autoreset=True)


def remove_links(markdown_text):
    # Replace [text](link) with text
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)


def ensure_ending_dot(text):
    if not text.strip().endswith('.'):
        return text.strip() + '.'
    else:
        return text.strip()


def parse_markdown_with_pauses(markdown_text, heading_pauses, max_length):

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
                    # Ensure ending dot for headings and list items
                    if is_heading or heading_level is not None:
                        sentence = ensure_ending_dot(sentence)
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
                # Ensure ending dot for headings and list items
                if is_heading or heading_level is not None:
                    sentence = ensure_ending_dot(sentence)
                segment = {
                    'text': sentence.strip(),
                    'pause_before': None,
                    'pause_after': None,
                    'is_heading': is_heading,
                    'heading_level': heading_level
                }
                segments.append(segment)
    return segments


def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


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


def similarity_ratio(a, b):
    matcher = SequenceMatcher(None, a, b)
    return matcher.ratio()


def calculate_wps(text, duration_seconds):
    word_count = len(text.strip().split())
    if duration_seconds > 0:
        return word_count / duration_seconds
    else:
        return 0


def generate_best_audio(text, args, stt_model, target_wps=None):
    print(f"{Fore.YELLOW}{Style.BRIGHT}Processing text:{Style.RESET_ALL} {text}")

    max_duration_seconds = 30.0  # Maximum allowed duration in seconds

    best_similarity = 0.0
    best_audio_path = None
    best_wps_difference = float('inf')
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
            'model_name': args.model_name,
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

        # Calculate WPS
        wps = calculate_wps(transcribed_text, generated_duration_seconds)
        print(f"Attempt {retries + 1}: WPS: {wps:.2f}")

        # If target WPS is not set, set it from the first successful attempt
        if target_wps is None and similarity >= args.min_similarity:
            target_wps = wps
            print(f"Estimated target WPS: {target_wps:.2f}")

        # Compute WPS difference
        if target_wps is not None:
            wps_difference = abs(wps - target_wps) / target_wps
        else:
            wps_difference = 0

        # Check if similarity and WPS are acceptable
        if similarity >= args.min_similarity and wps_difference <= args.wps_threshold:
            best_similarity = similarity
            best_audio_path = temp_path
            print("Audio accepted with acceptable similarity and WPS.")
            break

        # If this attempt is better, save it
        if (similarity > best_similarity) or (similarity == best_similarity and wps_difference < best_wps_difference):
            best_similarity = similarity
            best_wps_difference = wps_difference
            best_audio_path = temp_path
        else:
            # If no improvement, stop trying
            print("No improvement; stopping.")
            break

        # Adjust duration to get closer to target WPS
        if target_wps is not None:
            # Calculate desired duration to achieve target WPS
            word_count = len(text.strip().split())
            desired_duration = word_count / target_wps
            # Adjust duration_seconds towards desired duration
            if duration_seconds is None:
                duration_seconds = desired_duration
            else:
                # Average the current duration and desired duration
                duration_seconds = (duration_seconds + desired_duration) / 2
            # Ensure duration is within limits
            duration_seconds = min(duration_seconds, max_duration_seconds)
            if duration_seconds >= max_duration_seconds:
                print("Reached maximum duration limit.")
                break
            print(f"Adjusting duration to {duration_seconds:.2f} seconds and retrying...")
        else:
            # Increase duration by 20% for next attempt
            if duration_seconds is None:
                duration_seconds = min(generated_duration_seconds * 1.2, max_duration_seconds)
            else:
                duration_seconds = min(duration_seconds * 1.2, max_duration_seconds)
            if duration_seconds >= max_duration_seconds:
                print("Reached maximum duration limit.")
                break
            print(f"Increasing duration to {duration_seconds:.2f} seconds and retrying...")

        retries += 1

    # Load the best audio from the best attempt
    if best_audio_path is not None:
        audio, _ = sf.read(best_audio_path)
    else:
        # If no good audio was generated, use the last attempt
        print(f"Warning: Could not generate acceptable audio for text: {text}")
        audio, _ = sf.read(temp_path)

    # Clean up temporary files except the best one
    for file_path in temp_files:
        if os.path.exists(file_path) and file_path != best_audio_path:
            os.remove(file_path)

    return audio, target_wps


def text_to_audio(segments, args):
    # Initialize the speech-to-text model
    stt_model = Model('base.en', n_threads=4)  # Adjust n_threads as needed

    total_chunks = len(segments)
    start_time = time.time()

    sample_rate = 24000  # SAMPLE_RATE is 24000 Hz

    # Initialize target WPS
    target_wps = args.target_wps

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
                    audio, target_wps = generate_best_audio(text, args, stt_model, target_wps)
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


def convert_markdown_to_speech(markdown_text, output_file, **kwargs):
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    args = Args(**kwargs)
    args.output = output_file

    # Collect heading pauses from arguments
    heading_pauses = {
        'h1_before': getattr(args, 'pause_h1_before', 2.0),
        'h1_after': getattr(args, 'pause_h1_after', 0.7),
        'h2_before': getattr(args, 'pause_h2_before', 1.5),
        'h2_after': getattr(args, 'pause_h2_after', 0.7),
        'h3_before': getattr(args, 'pause_h3_before', 0.7),
        'h3_after': getattr(args, 'pause_h3_after', 0.7),
    }

    # Parse the markdown with pauses
    segments = parse_markdown_with_pauses(markdown_text, heading_pauses, args.max_length)

    # Generate audio and write incrementally
    text_to_audio(segments, args)
