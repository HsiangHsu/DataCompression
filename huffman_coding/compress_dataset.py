#!/usr/bin/env python3

from util import compress_file
import sys
import os

def main():
    if (len(sys.argv) < 2):
        sys.exit(f"Usage: {sys.argv[0]} infile [outfile]")

    infile = sys.argv[1]
    outfile = os.path.splitext(os.path.split(sys.argv[1])[1])[0] \
              + '_compressed.data'
    try:
        outfile = sys.argv[2]
    except IndexError:
        pass

    compress_file(infile, outfile)

if __name__ == "__main__":
    main()
