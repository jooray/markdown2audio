# Markdown to Speech Synthesizer

This script converts markdown files into spoken audio by generating speech for headings and paragraphs. It processes the markdown content, removes links, splits the text into manageable chunks, and synthesizes speech using the `f5-tts-mlx` module on Apple Silicon. The final audio is a concatenation of all generated speech segments, with optional pauses between them.

## Features

- Parses markdown files and extracts headings and paragraphs.
- Removes links while keeping the link text.
- Splits text into sentences and further into smaller chunks based on a configurable maximum length.
- Synthesizes speech for each text chunk.
- Concatenates generated audio segments with optional pauses.
- Supports optional reference audio for voice style matching.
- Allows setting a seed for reproducibility.
- Verifies the output audio with whisper speech to text, regenerates if the output has been cut off or is not good enough.

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
python markdown-to-audio.py <input_markdown_file.md> <output_audio_file.wav>
```

### Command Line Arguments

- **Positional Arguments:**
  - `input`: Input markdown file path.
  - `output`: Output audio file path.

- **Optional Arguments:**
  - `--ref-audio`: Reference audio file path.
  - `--ref-text`: Text spoken in the reference audio.
  - `--pause`: Pause duration in seconds between sentences (default: `0.5`).
  - `--max-length`: Maximum length of text chunk (default: `500` characters).
  - `--seed`: Seed for voice generation (default: `None`).

### Examples

#### Basic usage

```bash
python script.py input.md output.wav
```

#### With reference audio and text to clone voice

```bash
python script.py input.md output.wav --ref-audio path/to/ref_audio.wav --ref-text "Reference text."
```

Make sure that reference audio is 5-10 seconds long, ends with 1 second of silence.


You can convert an audio file to the correct format with ffmpeg like this:

```bash
ffmpeg -i /path/to/audio.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 /path/to/output_audio.wav
```

### Notes

- **Reference Audio:** Both `--ref-audio` and `--ref-text` must be provided together. If only one is supplied, the script will raise an error.
- **Temporary Files:** The script generates a temporary audio file (`temp_output.wav`) for each text chunk. Ensure you have write permissions in the working directory.
- **Cleanup:** The temporary files are not automatically deleted. You may want to remove them manually after execution.

