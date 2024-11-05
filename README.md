# Markdown to Speech Synthesizer

This script converts markdown files into spoken audio by generating speech for headings and paragraphs. It processes the markdown content, removes links, splits the text into manageable chunks, and synthesizes speech using the `f5-tts-mlx` module on Apple Silicon. The final audio is a concatenation of all generated speech segments, with optional pauses between them.

## Features

- Parses markdown files and extracts headings and paragraphs.
- Removes links while keeping the link text.
- Splits text into sentences and further into smaller chunks based on a configurable maximum length.
- Synthesizes speech for each text chunk, with options for maintaining a consistent speech rate across all generated audio.
- Concatenates generated audio segments with optional pauses.
- Supports optional reference audio for voice style matching.
- Allows setting a seed for reproducibility.
- Verifies the output audio with whisper speech-to-text and adjusts generation settings if needed for better quality and consistency.

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
  - `--ref-audio`: Reference audio file path.
  - `--ref-text`: Text spoken in the reference audio.
  - `--pause`: Pause duration in seconds between sentences (default: `0.5`).
  - `--max-length`: Maximum length of text chunk (default: `500` characters).
  - `--seed`: Seed for voice generation (default: `None`).
  - `--model-name`: Name of the TTS model to use (default: `lucasnewman/f5-tts-mlx`).
  - `--wps-threshold`: Allowed WPS (Words Per Second) variation threshold (default: `0.1` for 10%).
  - `--target-wps`: Target words per second. If not set, it is estimated from the first successful audio generation.
  - `--min-similarity`: Minimum acceptable similarity for generated audio (default: `0.9`).

- **Heading Pauses (Optional Arguments):**
  - `--pause-h1-before`: Pause before H1 headings in seconds (default: `2.0`).
  - `--pause-h1-after`: Pause after H1 headings in seconds (default: `0.7`).
  - `--pause-h2-before`: Pause before H2 headings in seconds (default: `1.5`).
  - `--pause-h2-after`: Pause after H2 headings in seconds (default: `0.7`).
  - `--pause-h3-before`: Pause before H3 headings in seconds (default: `0.7`).
  - `--pause-h3-after`: Pause after H3 headings in seconds (default: `0.7`).

### Examples

#### Basic usage

```bash
python -m markdown_to_speech.cli input.md output.wav
```

#### With reference audio and text to clone voice

```bash
python -m markdown_to_speech.cli input.md output.wav --ref-audio path/to/ref_audio.wav --ref-text "Reference text."
```

Make sure that reference audio is 5-10 seconds long, ending with 1 second of silence.

You can convert an audio file to the correct format with ffmpeg like this:

```bash
ffmpeg -i /path/to/audio.wav -ac 1 -ar 24000 -sample_fmt s16 -t 10 /path/to/output_audio.wav
```

#### Setting a Target Words Per Second (WPS)

```bash
python -m markdown_to_speech.cli input.md output.wav --target-wps 2.5
```

This sets a specific target WPS, ensuring consistent speech speed across all segments.

#### Adjusting WPS Threshold and Similarity

```bash
python -m markdown_to_speech.cli input.md output.wav --wps-threshold 0.05 --min-similarity 0.95
```

This sets a stricter WPS threshold of 5% and a higher minimum similarity of 95%.

## Explanation of WPS Consistency Feature

The script now includes a feature to maintain consistent Words Per Second (WPS) across all generated audio segments. This helps ensure that the pacing of the generated speech remains uniform throughout the entire output.

### How It Works

- For the **first segment**, the script generates audio and calculates the WPS.
  - If no `target-wps` is provided, this value becomes the target WPS for subsequent segments.
  - The WPS is printed to the console with the prefix: `"Estimated target WPS:"`.

- For **subsequent segments**, the script adjusts the duration to achieve the `target-wps`.
  - The script retries generation up to a maximum of 5 times if the WPS is not within the specified threshold (`--wps-threshold`, default is `0.1` for 10%).
  - The goal is to find an optimal combination of similarity (accuracy) and consistent pacing.

## Limitations and Considerations

- The **maximum duration** for any segment is capped at 30 seconds (`max-duration-seconds`) to avoid excessive generation times.
- The **WPS adjustment** is a balancing act between duration, similarity, and pacing. If similarity is below the acceptable level (`--min-similarity`), the script retries with different durations.
- The script outputs information about similarity and WPS for each generation attempt to help debug and optimize the pacing.

## License

This project is licensed under the GLWTS ("Good Luck With That Shit") License.

