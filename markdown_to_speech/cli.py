import argparse
from markdown_to_speech import convert_markdown_to_speech

def main():
    parser = argparse.ArgumentParser(description="Generate spoken audio from a markdown file.")
    parser.add_argument("input", help="Input markdown file path.")
    parser.add_argument("output", help="Output audio file path.")
    parser.add_argument("--ref-audio", default="", help="Reference audio file path.")
    parser.add_argument("--ref-text", default="", help="Text spoken in the reference audio.")
    parser.add_argument("--pause", type=float, default=0.5, help="Default pause duration in seconds between segments.")
    parser.add_argument("--max-length", type=int, default=500, help="Maximum length of text chunk.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for noise generation.")
    parser.add_argument("--model-name", type=str, default="lucasnewman/f5-tts-mlx", help="Name of the TTS model to use.")

    # Speaking pace (words per second) control
    parser.add_argument("--wps-threshold", type=float, default=0.1, help="Allowed WPS variation threshold (default: 0.1 for 10%).")
    parser.add_argument("--target-wps", type=float, default=None, help="Target words per second. If not set, estimated from the first audio.")
    parser.add_argument("--min-similarity", type=float, default=0.9, help="Minimum acceptable similarity (default: 0.9).")

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

    kwargs = vars(args)
    with open(args.input, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    convert_markdown_to_speech(markdown_text, args.output, **kwargs)

if __name__ == "__main__":
    main()
