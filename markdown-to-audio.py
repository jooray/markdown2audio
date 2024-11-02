import re
import argparse
import numpy as np
import soundfile as sf
import markdown
from bs4 import BeautifulSoup
from f5_tts_mlx.generate import generate
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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

# Function to generate audio from text
def text_to_audio(texts, args):
    audio_segments = []
    for text in texts:
        print("Generating: " + text)
        if not text.endswith('.'):
            text += '.'
        generate_args = {
            'generation_text': text,
            'model_name': 'lucasnewman/f5-tts-mlx',
            'output_path': 'temp_output.wav',
            'seed': args.seed
        }
        if args.ref_audio and args.ref_text:
            generate_args['ref_audio_path'] = args.ref_audio
            generate_args['ref_audio_text'] = args.ref_text
        elif args.ref_audio or args.ref_text:
            raise ValueError("Both reference audio and reference text must be provided together.")
        # Else, do not pass ref_audio_path and ref_audio_text

        generate(**generate_args)
        audio, _ = sf.read("temp_output.wav")
        audio_segments.append(audio)
        # Add pause
        if args.pause > 0:
            pause = np.zeros(int(args.pause * 24000))  # SAMPLE_RATE is 24000 Hz
            audio_segments.append(pause)

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
