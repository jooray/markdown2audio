import argparse
from markdown_to_speech import convert_markdown_to_speech

def main():
    parser = argparse.ArgumentParser(description="Generate spoken audio from a markdown file.")
    parser.add_argument("input", help="Input markdown file path.")
    parser.add_argument("output", help="Output audio file path.")
    parser.add_argument("--ref-audio", default="", help="Reference audio file path.")
    parser.add_argument("--pause", type=float, default=0.5, help="Default pause duration in seconds between segments.")

    # Add options for StyleTTS2 parameters
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha parameter for StyleTTS2 (default: 0.3).")
    parser.add_argument("--beta", type=float, default=0.7, help="Beta parameter for StyleTTS2 (default: 0.7).")
    parser.add_argument("--diffusion-steps", type=int, default=5, help="Diffusion steps for StyleTTS2 (default: 5).")
    parser.add_argument("--embedding-scale", type=float, default=1.0, help="Embedding scale for StyleTTS2 (default: 1.0).")

    # Similarity threshold
    parser.add_argument("--min-similarity", type=float, default=0.9, help="Minimum acceptable similarity (default: 0.9).")

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

    parser.add_argument('--split-at-headings', action='store_true', default=False,
                        help='Split text into segments at headings and paragraphs (default: False).')

    args = parser.parse_args()

    kwargs = vars(args)
    with open(args.input, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    convert_markdown_to_speech(markdown_text, args.output, **kwargs)

if __name__ == "__main__":
    main()
