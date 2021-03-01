# time python3 main.py --target 'quickdraw' --source 'quickdraw' --resnet152 --SGD --test --epochs 20 --bs 32
# time python3 main.py --target 'infograph' --source 'infograph' --resnet152 --SGD --test --epochs 20 --bs 32
# time python3 main.py --target 'sketch' --source 'sketch' --resnet152 --SGD --test --epochs 20 --bs 32

# time python3 main.py --target 'infograph' --source 'real' 'quickdraw' 'sketch' --resnet101 --SGD --test --epochs 25
# time python3 main.py --target 'quickdraw' --source 'infograph' 'real' 'sketch' --resnet101 --SGD --test --epochs 25
# time python3 main.py --target 'sketch' --source 'infograph' 'quickdraw' 'real' --resnet101 --SGD --test --epochs 25
# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet101 --SGD --test --epochs 25

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --SGD --test --lr 1e-3
# time python3 main.py --target 'infograph' --source 'real' 'quickdraw' 'sketch' --resnet152 --SGD --test --epochs 25
# time python3 main.py --target 'quickdraw' --source 'infograph' 'real' 'sketch' --resnet152 --SGD --test --epochs 25
# time python3 main.py --target 'sketch' --source 'infograph' 'quickdraw' 'real' --resnet152 --SGD --test --epochs 25

# time python3 main.py --target 'real' --source 'quickdraw' 'sketch' --resnet50 --SGD --test --epochs 25

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --inception --SGD --test --img_size 299 --epochs 25

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --SGD --test --epochs 25 --bs 64 # randomcrop

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --inception --SGD --test --img_size 299 --epochs 25 --resume --lr 1e-4

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --SGD --test --epochs 25 --bs 64 --resume --lr 1e-4 # randomcrop

# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --Adam --test --epochs 25 --bs 16

# time python3 main.py --target 'infograph' --source 'real' 'quickdraw' 'sketch' --inception --SGD --test --img_size 299 --epochs 25 --bs 64
# time python3 main.py --target 'quickdraw' --source 'infograph' 'real' 'sketch' --inception --SGD --test --img_size 299 --epochs 25 --bs 64
time python3 main.py --target 'sketch' --source 'infograph' 'quickdraw' 'real' --inception --SGD --test --img_size 299 --epochs 25
