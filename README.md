# Multi-source-Domain-Adaptation

## Task
A more practical scenario where training data are collected from multiple sources. For this challenge, we use the DomainNet dataset. We consider 3 source domains and 1 target domain, where each domain consists of 345 image classes. We will perform all 4 adaptations below. 
|   #   | Source Domains  | Target Domain |
| :---: | :-------------: | :-----------: |
|   1   | inf + qdr + rel |      skt      |
|   2   | inf +rel + skt  |      qdr      |
|   3   | qdr + rel + skt |      inf      |
|   4   | inf + qdr + skt |      rel      |

## Usage Format
Please run our code in the following manner:

    bash ./final.sh $1 $2

-   `$1` is the directory of dataset, which should contain folders of 'infograph', 'quickdraw', etc.
-   `$2` is the directory to place your output prediction files (e.g. `output_infograph.csv`, `output_quickdraw.csv`, `output_sketch.csv`, `output_real.csv`).
