# MDAN Final Models

## Experimental Results
- MDAN on the testing dataset.

|  Target domains  |   skt   |   qdr   |   inf   |   rel   |
| :--------------: | :-----: | :-----: | :-----: | :-----: |
|       MDAN       |  50.1%  |  17.4%  |  24.0%  |  57.0%  |

The column-wise domains are selected as the target domain.
(inf: infograph, qdr: quickdraw, skt: sketch, rel: real)

## Models
Please download the models from dropbox links below:

-   [infograph](https://www.dropbox.com/s/dhynuiha5i0sjjc/res152_MDAN-infograph.pth?dl=1)
-   [quickdraw](https://www.dropbox.com/s/2zzag5j81gx1znt/res152_MDAN-quickdraw.pth?dl=1)
-   [real](https://www.dropbox.com/s/qabfnc6idm4zdel/res152_MDAN-real.pth?dl=1)
-   [sketch](https://www.dropbox.com/s/yq29sprzkffrofw/res152_MDAN-sketch.pth?dl=1)

## Usage Format
Please run our code in the following manner:

    bash ./test4domain.sh 

-   Please change the data and model directory in the shell script first.
-   Please choose the target domain in number 0,1,2,3. (inf: 0, qdr: 1, rel: 2, skt: 3)

## Citation
    @article{zhao2018multiple,
      title={Multiple source domain adaptation with adversarial learning},
      author={Zhao, Han and Zhang, Shanghang and Wu, Guanhang and Moura, Jos{\'e} MF and Costeira, Joao P and Gordon, Geoffrey J},
      booktitle={International Conference on Learning Representations, workshop track},
      year={2018}
    }
