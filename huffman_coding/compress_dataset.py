#!/usr/bin/env python3

from util.compressor import compress_file
import sys
import os
from timeit import default_timer as timer
from datetime import timedelta
from warnings import filterwarnings

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

    filterwarnings('ignore')

    start = timer()
    compress_file(infile, outfile)
    end = timer()

    print(f"Compression completed in {timedelta(seconds=(round(end - start)))}.")

if __name__ == "__main__":
    main()
