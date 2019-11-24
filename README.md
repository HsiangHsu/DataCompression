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

| Compression  | Size   |
|:------------:|:------:|
| Uncompressed | 4 MB   |
| ZIP          | 441 KB |
| Huffman      | 74 KB  |
