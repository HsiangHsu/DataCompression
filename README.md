# DataCompression

## Huffman Coding
Install dependencies:
```
pip3 install bitstring
```

Compress a dataset:
```
./compress_dataset.py ../datasets/adult/adult.data
```

Decompress a dataset:
```
./decompress_dataset.py adult_compressed.data
```

---

Statistics on Adult Data Set:

| Compression             | Size   |
|:-----------------------:|:------:|
| Uncompressed            | 4 MB   |
| ZIP                     | 441 KB |
| Huffman (Discrete Only) | 74 KB  |
| Huffman                 | 139 KB |

Continuous fields are quantized prior to encoding using a Lloyd-Max quantizer
using 16 bins. Lloyd-Max is run over each category 64 times to try and
experimentally minimize the MSE. The exception is 'fnlwgt', on which Lloyd-Max
is only run once, as it takes so long to converge for that category. The
runtime for compression on the Adult dataset is around 3 minutes. The vast
majority of this time is spent running Lloyd-Max iterations. The runtime for
decompression is around 1 second.

Although it varies a bit from run to run, the MSEs for the quantized fields
tend to be around the following values, taken from a single example trial:

| Field          | MSE            |
|:--------------:|:--------------:|
| age            | 1.45           |
| fnlwgt         | 124,134,121.10 |
| education-num  | 0.00           |
| capital-gain   | 11,999.28      |
| capital-loss   | 57.36          |
| hours-per-week | 0.89           |

Although 'education-num' is technically classified as a continuous field,
because it only takes on 16 values, it is functionally a discrete field,
and therefore the MSE is always 0.
