# Markdown to Speech Synthesizer

This script converts markdown files into spoken audio by generating speech for headings and paragraphs. It processes the markdown content, removes links and formatting, and synthesizes speech using the `styletts2` module. The final audio is a concatenation of all generated speech segments, with optional pauses between them.

## Features

- Parses markdown files and extracts headings and paragraphs.
- Removes links while keeping the link text.
- Processes text either as one whole chunk or splits into paragraphs and headings based on the `--split-at-headings` parameter.
- Keeps paragraphs intact without splitting sentences inside.
- Synthesizes speech for each text chunk, with options for adjusting voice style parameters.
- Concatenates generated audio segments with optional pauses.
- Supports optional reference audio for voice cloning.
- Allows setting parameters for `alpha`, `beta`, `diffusion_steps`, and `embedding_scale`.
- Verifies the output audio with Whisper speech-to-text and adjusts generation settings if needed for better quality and consistency.

## Installation

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/markdown-to-speech.git
   cd markdown-to-speech
   ```

2. **Install Dependencies**

   Using Poetry:

   ```bash
   poetry install
   ```

   This will create a virtual environment and install all required dependencies as specified in the `pyproject.toml` file.

## Usage

Activate the virtual environment (if not already activated):

```bash
poetry shell
```

Run the script with the required positional arguments for input and output files:

```bash
python -m markdown_to_speech.cli <input_markdown_file.md> <output_audio_file.wav>
```

### Command Line Arguments

- **Positional Arguments:**
  - `input`: Input markdown file path.
  - `output`: Output audio file path.

- **Optional Arguments:**
  - `--ref-audio`: Reference audio file path for voice cloning.
  - `--pause`: Default pause duration in seconds between segments (default: `0.5`).
  - `--split-at-headings`: Split text into segments at headings and paragraphs (default: `False`).

- **StyleTTS2 Parameters:**
  - `--alpha`: Alpha parameter for StyleTTS2 (default: `0.3`).
  - `--beta`: Beta parameter for StyleTTS2 (default: `0.7`).
  - `--diffusion-steps`: Diffusion steps for StyleTTS2 (default: `5`).
  - `--embedding-scale`: Embedding scale for StyleTTS2 (default: `1.0`).

- **Similarity Parameter:**
  - `--min-similarity`: Minimum acceptable similarity for generated audio (default: `0.9`).

- **Heading Pauses (Optional Arguments):**
  - `--pause-h1-before`: Pause before H1 headings in seconds (default: `2.0`).
  - `--pause-h1-after`: Pause after H1 headings in seconds (default: `0.7`).
  - `--pause-h2-before`: Pause before H2 headings in seconds (default: `1.5`).
  - `--pause-h2-after`: Pause after H2 headings in seconds (default: `0.7`).
  - `--pause-h3-before`: Pause before H3 headings in seconds (default: `0.7`).
  - `--pause-h3-after`: Pause after H3 headings in seconds (default: `0.7`).

### Examples

#### Basic usage without splitting (process entire text as one chunk)

```bash
python -m markdown_to_speech.cli input.md output.wav
```

#### Splitting at headings and paragraphs

```bash
python -m markdown_to_speech.cli input.md output.wav --split-at-headings
```

#### With reference audio to clone voice

```bash
python -m markdown_to_speech.cli input.md output.wav --ref-audio path/to/ref_audio.wav
```

Ensure that the reference audio file exists and is in a compatible format.

#### Adjusting StyleTTS2 Parameters

```bash
python -m markdown_to_speech.cli input.md output.wav --alpha 0.5 --beta 0.5 --diffusion-steps 7 --embedding-scale 1.2
```

This sets specific parameters for the StyleTTS2 model to fine-tune the voice characteristics.

## License

This project is licensed under the GLWTS ("Good Luck With That Shit") License.
