time python3 main_dann.py  --target 'real' --source 'infograph' 'quickdraw' 'sketch' --model resnet152 --train --r
#time python3 main_dann.py  --target 'infograph' --source 'real' 'quickdraw' 'sketch' --model resnet152 --train --r --resume_path models/pretrained/resnet152_infograph_separated.pth
#time python3 main_dann.py  --target 'quickdraw' --source 'infograph' 'real' 'sketch' --model resnet152 --train --r --resume_path models/pretrained/resnet152_quickdraw_separated.pth
#time python3 main_dann.py  --target 'sketch' --source 'infograph' 'quickdraw' 'real' --model resnet152 --train --r --resume_path models/pretrained/resnet152_sketch_separated.pth
