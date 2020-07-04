import argparse
import jpeg_ls
import os

from loader_cp import load

parser = argparse.ArgumentParser()
parser.add_argument('d', type=str, help='dataset to benchmark',
    choices=['mnist', 'cifar-10'])
args = parser.parse_args()

data = load(args.d)

print(data.shape)

original_size = data.size
total_enc_size = 0

for image in data:
    enc_image = jpeg_ls.encode(image)
    total_enc_size += enc_image.size

print(f'Original size: {original_size/1000000:.1f} MB')
print(f'Total encoded size: {total_enc_size/1000000:.1f} MB')
print(f'Compression ratio: {total_enc_size/original_size:.2f}')
