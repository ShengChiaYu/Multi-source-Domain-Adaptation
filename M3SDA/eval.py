import numpy as np
import pandas as pd
import sys

pred_csv = sys.argv[1]
gt_csv = sys.argv[2]

pred_data = pd.read_csv(pred_csv)
pred = np.array(pred_data['label'])

gt_data = pd.read_csv(gt_csv)
gt = np.array(gt_data['label'])

acc = float(sum(pred == gt)) / float(len(pred))
print ('acc = {:4f}'.format(acc))
