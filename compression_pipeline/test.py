import argparse

parser = argparse.ArgumentParser()
parser.add_argument('square', type=int,
    help='display a square of a given number')
parser.add_argument('-v', '--verbosity', help='increase output verbosity',
    type=int)
args = parser.parse_args()
answer = args.square**2
if args.verbosity == 2:
    print(f'the square of {args.square} equals {answer}')
else:
    print(answer)

