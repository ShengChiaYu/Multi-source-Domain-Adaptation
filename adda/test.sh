# time python3 predict.py --model_name 'src_infograph.pth' --target 'infograph' --resnet152 1
time python3 predict.py --model_name 'src_infograph.pth' --target 'quickdraw' --resnet152 1
time python3 predict.py --model_name 'src_infograph.pth' --target 'sketch' --resnet152 1

time python3 predict.py --model_name 'src_quickdraw.pth' --target 'infograph' --resnet152 1
time python3 predict.py --model_name 'src_quickdraw.pth' --target 'quickdraw' --resnet152 1
time python3 predict.py --model_name 'src_quickdraw.pth' --target 'sketch' --resnet152 1

time python3 predict.py --model_name 'src_sketch.pth' --target 'infograph' --resnet152 1
time python3 predict.py --model_name 'src_sketch.pth' --target 'quickdraw' --resnet152 1
time python3 predict.py --model_name 'src_sketch.pth' --target 'sketch' --resnet152 1

# time python3 predict.py --model_name 'src_combine_tar_real.pth' --resnet101 1 --real_test 1
