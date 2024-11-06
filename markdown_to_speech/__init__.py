import re
import numpy as np
import soundfile as sf
from pywhispercpp.model import Model
from difflib import SequenceMatcher
import os
import tempfile
import time
from tqdm import tqdm
from colorama import init, Fore, Style
from styletts2 import tts  # Import the StyleTTS2 model

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


def remove_markdown_formatting(text):
    # Remove markdown formatting
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove heading markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italics
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)     # Images
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Inline code
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
    text = re.sub(r'>\s*(.*)', r'\1', text)         # Blockquotes
    text = re.sub(r'-\s*(.*)', r'\1', text)         # Lists
    text = re.sub(r'\n\s*\n', '\n', text)           # Remove multiple newlines
    text = re.sub(r'”', '"', text)   # Replace smart quotes
    return text


def parse_markdown(markdown_text, heading_pauses, split_at_headings):
    # Remove links
    markdown_text = remove_links(markdown_text)

    # Split the markdown text into lines
    lines = markdown_text.split('\n')
    processed_lines = []
    in_list_item = False
    list_item_indent = 0
    current_paragraph = []

    def process_paragraph():
        if current_paragraph:
            current_paragraph[-1] = ensure_ending_dot(current_paragraph[-1])
            processed_lines.extend(current_paragraph)
            current_paragraph.clear()

    for i, line in enumerate(lines):
        stripped_line = line.strip()

        # Check if the line is a heading
        if stripped_line.startswith('#'):
            process_paragraph()
            processed_lines.append(ensure_ending_dot(stripped_line))
            in_list_item = False
        # Check if the line is the start of a list item
        elif re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
            process_paragraph()
            current_paragraph.append(stripped_line)
            in_list_item = True
            list_item_indent = len(line) - len(line.lstrip())
        elif in_list_item and len(line) - len(line.lstrip()) > list_item_indent:
            # This is a continuation of a list item
            current_paragraph.append(stripped_line)
        elif stripped_line:
            # It's a non-empty line (start of a new paragraph or continuation)
            if not current_paragraph:
                # Start of a new paragraph
                current_paragraph.append(stripped_line)
            else:
                # Continuation of the current paragraph
                current_paragraph[-1] += ' ' + stripped_line
            in_list_item = False
        else:
            # Empty line, end of paragraph
            process_paragraph()
            in_list_item = False

    # Process any remaining paragraph
    process_paragraph()

    processed_text = '\n'.join(processed_lines)

    segments = []

    if not split_at_headings:
        # Remove markdown formatting
        text = remove_markdown_formatting(processed_text)
        if text:
            segments.append({'text': text.strip(), 'is_heading': False})
        return segments


    else:
        # Process paragraph by paragraph
        paragraph = ''
        for line in lines:
            line = line.rstrip()
            if not line:
                # Empty line indicates end of paragraph
                if paragraph:
                    # Process paragraph
                    segment = process_paragraph(paragraph, heading_pauses)
                    segments.extend(segment)
                    paragraph = ''
            else:
                paragraph += line + '\n'

        # Process any remaining paragraph
        if paragraph:
            segment = process_paragraph(paragraph, heading_pauses)
            segments.extend(segment)

        return segments


def process_paragraph(paragraph, heading_pauses):
    paragraph = paragraph.strip()
    if not paragraph:
        return []
    # Check if paragraph is a heading
    heading_match = re.match(r'^(#{1,6})\s*(.*)', paragraph)
    if heading_match:
        hashes, heading_text = heading_match.groups()
        level = min(len(hashes), 3)
        pause_before = heading_pauses.get(f'h{level}_before', 0)
        pause_after = heading_pauses.get(f'h{level}_after', 0)
        is_heading = True
        heading_level = level
        # Remove markdown formatting
        heading_text = remove_markdown_formatting(heading_text)
        # Ensure ending dot
        heading_text = ensure_ending_dot(heading_text)
        segment = {
            'text': heading_text,
            'pause_before': pause_before,
            'pause_after': pause_after,
            'is_heading': True,
            'heading_level': heading_level
        }
        return [segment]
    else:
        # Regular paragraph
        # Remove markdown formatting
        paragraph_text = remove_markdown_formatting(paragraph)
        paragraph_text = ensure_ending_dot(paragraph_text)
        segment = {
            'text': paragraph_text,
            'pause_before': None,
            'pause_after': None,
            'is_heading': False
        }
        return [segment]


def similarity_ratio(a, b):
    matcher = SequenceMatcher(None, a, b)
    return matcher.ratio()

def generate_best_audio(text, args, stt_model, tts_model, ref_s=None):
    print(f"{Fore.YELLOW}{Style.BRIGHT}Processing text:{Style.RESET_ALL} {text}")

    max_retries = 5 if args.max_retries is None else args.max_retries
    retries = 0
    best_similarity = 0.0
    best_audio = None

    while retries < max_retries:
        # Adjust diffusion_steps on each retry to improve similarity
        diffusion_steps = args.diffusion_steps + retries  # Increase diffusion_steps on each retry

        # Create output WAV file path
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)

        # Generate audio
        try:
            audio = tts_model.inference(
                text=text,
                ref_s=ref_s,
                output_sample_rate=24000,
                alpha=args.alpha,
                beta=args.beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=args.embedding_scale,
                output_wav_file=temp_path
            )
        except Exception as e:
            print(f"Error during inference: {e}")
            os.remove(temp_path)
            break

        # Read generated audio to get its duration
        audio_data, _ = sf.read(temp_path)

        # Transcribe the audio
        segments = stt_model.transcribe(temp_path)
        transcribed_text = ''.join([segment.text for segment in segments]).strip()

        # Output transcription for debugging
        print(f"{Fore.YELLOW}{Style.BRIGHT}Transcribed text:{Style.RESET_ALL} {transcribed_text}")

        # Clean up transcribed text and original text for comparison
        clean_transcribed_text = re.sub(r'[\s.,\'"\"-]+', ' ', transcribed_text.lower()).strip()
        clean_original_text = re.sub(r'[\s.,\'"\"-]+', ' ', text.lower()).strip()

        # Compute similarity of the whole text
        similarity = similarity_ratio(clean_transcribed_text, clean_original_text)
        print(f"Attempt {retries + 1}: Similarity: {similarity:.2f}")

        # Check if this attempt's similarity is the best so far
        if similarity > best_similarity:
            best_similarity = similarity
            best_audio = audio_data  # Update best audio with highest similarity found so far
            print("New best audio found with updated similarity.")

        # Stop searching if similarity meets or exceeds the minimum threshold
        if similarity >= args.min_similarity:
            print("Audio accepted with acceptable similarity.")
            os.remove(temp_path)
            break
        else:
            print("Similarity below threshold, retrying with increased diffusion_steps.")
            retries += 1
            os.remove(temp_path)

    if best_audio is None:
        print(f"Warning: Could not generate acceptable audio for text: {text}")
        best_audio = np.zeros(int(0.5 * 24000))  # Insert silence as a placeholder

    return best_audio



def text_to_audio(segments, args):
    # Initialize the speech-to-text model
    stt_model = Model('base.en', n_threads=4)  # Adjust n_threads as needed

    # Initialize the TTS model
    tts_model = tts.StyleTTS2(phoneme_converter='espeak')

    # Precompute the target voice vector if a target voice is specified
    ref_s = None
    if args.ref_audio:
        if os.path.exists(args.ref_audio):
            print("Computing style vector from reference audio...")
            ref_s = tts_model.compute_style(args.ref_audio)
        else:
            print(f"Reference audio file not found: {args.ref_audio}")

    total_chunks = len(segments)
    start_time = time.time()

    sample_rate = 24000

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
                    audio = generate_best_audio(text, args, stt_model, tts_model, ref_s=ref_s)
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

    # Parse the markdown
    segments = parse_markdown(markdown_text, heading_pauses, args.split_at_headings)

    # Generate audio and write incrementally
    text_to_audio(segments, args)
