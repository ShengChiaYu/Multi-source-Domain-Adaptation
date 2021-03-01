# time python3 predict.py --model_name 'src_infograph.pth' --target 'infograph' --resnet152
# time python3 predict.py --model_name 'src_infograph.pth' --target 'quickdraw' --resnet152
# time python3 predict.py --model_name 'src_infograph.pth' --target 'sketch' --resnet152
#
# time python3 predict.py --model_name 'src_quickdraw.pth' --target 'infograph' --resnet152
# time python3 predict.py --model_name 'src_quickdraw.pth' --target 'quickdraw' --resnet152
# time python3 predict.py --model_name 'src_quickdraw.pth' --target 'sketch' --resnet152
#
# time python3 predict.py --model_name 'src_sketch.pth' --target 'infograph' --resnet152
# time python3 predict.py --model_name 'src_sketch.pth' --target 'quickdraw' --resnet152
# time python3 predict.py --model_name 'src_sketch.pth' --target 'sketch' --resnet152

time python3 predict.py --model_name 'src_combine_tar_infograph.pth' --target 'infograph' --inception --img_size 299
time python3 predict.py --model_name 'src_combine_tar_quickdraw.pth' --target 'quickdraw' --inception --img_size 299
time python3 predict.py --model_name 'src_combine_tar_sketch.pth' --target 'sketch' --inception --img_size 299 
