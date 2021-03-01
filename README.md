# Final Project

## Experimental Results
- M3SDA on the testing dataset.

|  Target domains  |   skt   |   qdr   |   inf   |   rel   |
| :--------------: | :-----: | :-----: | :-----: | :-----: |
|      M3SDA       | 48.545% | 16.609% | 24.451% | 57.690% |

The column-wise domains are selected as the target domain.
(inf: infograph, qdr: quickdraw, skt: sketch, rel: real)

## Usage Format
Please run our code in the following manner:

    bash ./final.sh $1 $2

-   `$1` is the directory of dataset, which should contain folders of 'infograph', 'quickdraw', etc.
-   `$2` is the directory to place your output prediction files (e.g. `output_infograph.csv`, `output_quickdraw.csv`, `output_sketch.csv`, `output_real.csv`).
