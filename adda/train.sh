# time python3 main.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --SGD --bs 64 --lr 1e-4 --epochs 25
# time python3 main.py --target 'infograph' --source 'real' 'quickdraw' 'sketch' --resnet101 --extractor --SGD --epochs 25
# time python3 main.py --target 'quickdraw' --source 'infograph' 'real' 'sketch' --resnet101 --extractor --SGD --epochs 25
# time python3 main.py --target 'sketch' --source 'infograph' 'quickdraw' 'real' --resnet101 --extractor --SGD --epochs 25

time python3 adda.py --target 'real' --source 'infograph' 'quickdraw' 'sketch' --resnet152 --SGD --bs 8 --lr 1e-4 --epochs 25
