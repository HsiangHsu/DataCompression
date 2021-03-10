# DataCompression

The compression pipeline has four stages:

- Load
- Preprocess
- Compress
- Encode

The high-level logic for each of them is found in `driver.py` and is called by `compress.py`. Each subdirectory contains specific implementations of different preprocessors, compressors, and encoders.

The _load_ stage unpacks the desired dataset into a properly shaped NumPy array. The _preprocess_ stage computes any needed data for compression; in predictive coding, this incorporates extraction of training context  and training the model. The _compress_ stage goes item-by-item in the dataset and applies some sort of compression logic, whether that's reordering to minimize inter-element distance or applying the predictive model and building the error string. The _encode_ stage is entropy coding; we support both Huffman and Golomb coding.

## Predictive Coding
Install dependencies:
```
pip3 install -r requirements.txt 
```

### Compress a dataset:
```
python3 compress.py [DATASET] --pre predictive --ordering [ORDER] --prev-context [PREV]
--current-context [CURR] --comp predictive --enc pred-huff --num-prev-imgs [N] 
--predictor-family [MODEL] --mode [RGB MODE]
```
Parameters:

| Name                    | Meaning   |
|:-----------------------:|:------:|
| DATASET            | one of mnist, adult, cifar-10, test, synthetic   |
| ORDER | Either mst (minimum spanning tree) or random  |
| PREV                 | One of DAB, DABC, DABX |
| CURR                 | One of DAB, DABC |
| MODEL                 | One of linear, logistic, [cubist](https://cran.r-project.org/web/packages/Cubist/vignettes/cubist.html), quantile (ML predicting which quantile the pixel falls into) |
| RGB MODE                 | Either single (to train a single predictor for rgb tuples) or triple (for three separate predictors); logistic models only work with triple mode |

You can also pass in `--feature-file` and `--label-file` to use pre-extracted training context or `--predictor-file` to use an existing predictor. See compress.py for additional optional parameters. 

### Decompress a dataset:
```
./decompress.py --data_in [comp.out FILE] --args_in [args.out FILE]
```
Then run
```
./verify.py
```
to check that the decompressed dataset equals the original.

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

---

## MNIST Naive Tests

Statistics on MNIST Data Set:

| Compression                     | Size   |
|:-------------------------------:|:------:|
| Uncompressed                    | 47 MB  |
| ZIP                             | 9.8 MB |
| B/W Bits                        | 5.9 MB |
| Row Range Encode Differences    | 4.3 MB |
| Column Range Encode Differences | 3.9 MB |

B/W Bits compression consists of quantizing every non-zero greyscale pixel
as a 1 and then encoding each pixel as a bit instead of a byte. Thus,
it simply cuts down on the file size by a factor of 8.

Row Range Encode Differences then calculates the "average" image for each
label by determing whether a given pixel is a 0 or a 1 for a majority of the images
in the label. The difference bitmap for each image is calculated relative to
this average, with a 0 indicating the same pixel as the average at that location
and a 1 indicating a flipped pixel. To compress this difference representation
to a more realistically encodeable form, we write out each row from the
difference image as a series of half-open ranges that indicate what pixels are
flipped. For example, if a given row in the difference image had a 1 for pixels
in columns `[0, 1, 2, 7, 8, 9]`, then this would be shortened to
`[0, 3, 7, 10]`. It is this shortened array that is then stored directly (with each value
being written as a 5-bit number, which is possible since the images are only 28x28).

Column Range Encode Differences runs a similar process, except the transpose of the
difference bitmap is taken before encoding, such that the difference ranges are
recorded along columns instead of rows.

