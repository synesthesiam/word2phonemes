import os
import sys
import argparse
import itertools
import logging

from .model import WordToPhonemeModel

def main():
    parser = argparse.ArgumentParser('word2phonemes')
    parser.add_argument('words', nargs="*", type=str,
                        help='Words to guess pronunciations for')
    parser.add_argument('--language', type=str, default='en',
                        help='Model language')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory with models')
    parser.add_argument('--syllables', type=int, default=0,
                        help='Split words into groups of N syllables first (N=0 is disabled)')
    parser.add_argument('--syllable-language', type=str, default=None,
                        help='Language to use for syllable splitting (default=auto)')
    parser.add_argument('--train', type=int, default=0,
                        help='Train for N epochs')

    args = parser.parse_args()
    model_dir = os.path.join(args.data_dir, args.language)

    # -------------------------------------------------------------------------

    if args.train > 0:
        logging.basicConfig(level=logging.DEBUG)

        model = WordToPhonemeModel()
        model.load_dataset(os.path.join(model_dir, 'dictionary.csv'))
        model.train(args.train, os.path.join(model_dir, 'g2p-model.pt'))
        sys.exit(0)

    # -------------------------------------------------------------------------

    model = WordToPhonemeModel(model_dir)

    words = args.words if len(args.words) > 0 else sys.stdin
    hyph = None

    for word in words:
        word = word.strip()

        if args.syllables > 0:
            # Split words into groups of syllables
            if hyph is None:
                from hyphen import Hyphenator
                hyph = Hyphenator(args.language, directory=model_dir)

            syllables = hyph.syllables(word)
            groups = [''.join(g) for g in grouper(syllables, args.syllables, '')]

            # Join syllable group pronunciations
            phonemes = [' '.join(model.word2phonemes(g)) for g in groups]
        else:
            # Use entire word
            phonemes = model.word2phonemes(word)

        print(word, ' '.join(phonemes))

# -----------------------------------------------------------------------------

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
