import os
import sys
import argparse
import itertools
import logging
import time

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
    parser.add_argument('--phonemes-upper', action='store_true',
                        help='Uppercase phonemes before printing')
    parser.add_argument('--debug', action='store_true',
                        help='Print DEBUG messages')

    args = parser.parse_args()
    model_dir = os.path.join(args.data_dir, args.language)

    # -------------------------------------------------------------------------

    # Do training
    if args.train > 0:
        logging.basicConfig(level=logging.DEBUG)

        save_path = os.path.abspath(os.path.join(model_dir, 'g2p-model.pt'))
        model = WordToPhonemeModel()

        start_time = time.time()
        model.load_dataset(os.path.join(model_dir, 'dictionary.csv'))
        model.save_vocabulary(model_dir)
        model.train(args.train, save_path)
        train_time = time.time() - start_time

        logging.info(f'Training completed in {train_time} second(s)')
        logging.debug(f'Wrote {save_path}')

        return

    # -------------------------------------------------------------------------

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Do inference
    model = WordToPhonemeModel(model_dir)

    # Get words from command-line or stdin
    words = args.words if len(args.words) > 0 else sys.stdin
    hyph = None

    for word in words:
        word = word.strip()
        logging.debug(f'Processing {word}')

        if args.syllables > 0:
            # Split words into groups of syllables
            if hyph is None:
                from hyphen import Hyphenator
                logging.debug(f'Loading hyphenator for {args.language} from {model_dir}')
                hyph = Hyphenator(args.language, directory=model_dir)

            syllables = hyph.syllables(word)
            groups = [''.join(g) for g in grouper(syllables, args.syllables, '')]
            logging.debug(f'Syllable groups: {groups}')

            # Join syllable group pronunciations
            phonemes = [' '.join(model.word2phonemes(g)) for g in groups]
        else:
            # Use entire word
            phonemes = model.word2phonemes(word)


        # Print result
        phonemes_str = ' '.join(phonemes)
        if args.phonemes_upper:
            phonemes_str = phonemes_str.upper()

        print(word, phonemes_str)

# -----------------------------------------------------------------------------

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
