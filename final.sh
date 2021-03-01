wget -O m3sda_real.pth 'https://www.dropbox.com/s/n6fpzaanbiitm3a/m3sda_real_2_0.5734502446982055.pth?dl=1'
wget -O m3sda_infograph.pth 'https://www.dropbox.com/s/0q5n08zi9209z3g/m3sda_infograph_1_0.2481132075471698.pth?dl=1'
wget -O m3sda_quickdraw.pth 'https://www.dropbox.com/s/375xpa7kdz6a37t/m3sda_quickdraw_1_0.16392764002152585.pth?dl=1'
wget -O m3sda_sketch.pth 'https://www.dropbox.com/s/qso0g613xnrj6ja/m3sda_sketch_2_0.4826007326007326.pth?dl=1'

mv m3sda_real.pth M3SDA/
mv m3sda_infograph.pth M3SDA/
mv m3sda_quickdraw.pth M3SDA/
mv m3sda_sketch.pth M3SDA/

time python3 M3SDA/main.py --resnet152 1 --target 'real' --title 'test' --model_path M3SDA/m3sda_real.pth --data_dir $1 --pred_dir $2 --test
time python3 M3SDA/main.py --resnet152 1 --target 'infograph' --title 'infograph' --model_path M3SDA/m3sda_infograph.pth --data_dir $1 --pred_dir $2 --test
time python3 M3SDA/main.py --resnet152 1 --target 'quickdraw' --title 'quickdraw' --model_path M3SDA/m3sda_quickdraw.pth --data_dir $1 --pred_dir $2 --test
time python3 M3SDA/main.py --resnet152 1 --target 'sketch' --title 'sketch' --model_path M3SDA/m3sda_sketch.pth --data_dir $1 --pred_dir $2 --test
