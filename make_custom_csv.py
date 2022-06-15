import os
import shutil
import pandas as pd

from datasets.dataset import class_dict
from utils.utils import make_mask


cropped = 'carpal'
mode = 'valid'

data_path = f'data/{cropped}/{mode}'
fold_lst = sorted(os.listdir(data_path), key=lambda x: float(x))
print(data_path)
# stop = [1.5, 6.0, 8.5, 11.0, 13.0, 15.5, 18.5]  # ulna
# stop = [1.5, 5.5, 8.5, 12.5, 16.5, 18.5] # pp3 8/6/8/8/5 // 1,2,3,4,5
# stop = [1.5, 6.0, 10.0, 12.5, 15.0, 18.5] # mp3 1.5 2 2.5 3 3.5 4 4.5 5 5.5
# stop = [1.5, 7.0, 12.0, 16.0, 18.5] # dp3
# stop = [1.5, 2.5, 5.5, 9.0, 13.0, 14.5, 18.5] # radius
# stop = [1.5, 5.0, 9.5, 13.0, 15.5, 18.5] # mc1
stop = [1.5, 5.5, 11.5, 15, 18.5] # carpal



img_lst = []
main_lst = []
sub_lst = []
for i, (s1, s2) in enumerate(zip(stop[:-1], stop[1:])):
    for fold in fold_lst:
        if s1 <= float(fold) < s2:
            file_lst = os.listdir(os.path.join(data_path, fold))
            for file in file_lst:
                img_lst.append(os.path.join(data_path, fold, file))
                main_lst.append(i)
                for k, v in class_dict.items():
                    if v == float(fold):
                        sub_lst.append(k)
        else:
            continue

df = pd.DataFrame({'img_path': img_lst, 'main': main_lst, 'sub': sub_lst})
df.to_csv(f'data/{cropped}_{mode}.csv', index=False)

# example ulna
stop_lst = [k for s in stop for k, v in class_dict.items() if v == s]
mask = make_mask(stop_lst)
print(mask)
