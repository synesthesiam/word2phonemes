#!/usr/bin/env python3
import sys
import re

# Takes a CMU dictionary as input.
# Outputs a CSV file with only words that have alpha characters and
# apostrophes.

def main():
    for line in sys.stdin:
        line = line.strip()
        if len(line) == 0:
            continue

        # test T EH S T
        word, phonemes = re.split(r'\s+', line, maxsplit=1)

        # Remove pronunciation number: a(2) -> a
        if '(' in word:
            word = word[:word.index('(')]

        if all(c.isalpha() or (c in ["'"]) for c in word):
            # test,T EH S T
            print(f'{word},{phonemes}')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
