# time python3 main.py --target 'quickdraw' --source 'quickdraw' --resnet152 1 --SGD 1 --test 1 --epochs 20 --bs 32
# time python3 main.py --target 'infograph' --source 'infograph' --resnet152 1 --SGD 1 --test 1 --epochs 20 --bs 32
# time python3 main.py --target 'sketch' --source 'sketch' --resnet152 1 --SGD 1 --test 1 --epochs 20 --bs 32
# time python3 main.py --target 'real' --source 'real' --resnet152 1 --SGD 1 --epochs 20 --bs 32

time python3 main.py --target 'infograph' --source 'real' 'quickdraw' 'sketch' --resnet152 1 --SGD 1 --train --model_name 'src_combine_tar_infograph_separated.pth'
time python3 main.py --target 'quickdraw' --source 'infograph' 'real' 'sketch' --resnet152 1 --SGD 1 --train --model_name 'src_combine_tar_quickdraw_separated.pth'
time python3 main.py --target 'sketch' --source 'infograph' 'quickdraw' 'real' --resnet152 1 --SGD 1 --train --model_name 'src_combine_tar_sketch_separated.pth'
# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 1 --SGD 1 --train --bs 8 --lr 1e-4

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 1 --SGD 1 --test
