import os
from os.path import join
from os import listdir
import random
import shutil

my_path = 'ufi-cropped\\train\\'
folders = [f for f in listdir(my_path)]

files = []
for folder_name in folders:
    folder_path = join(my_path, folder_name)
    files.append([f for f in listdir(folder_path) if f.endswith('.pgm')])

train_set = []
for fo in range(len(folders)):
    pathname_list = []
    for fi in files[fo]:
        pathname_list.append(join(my_path, join(folders[fo], fi)))
    train_set.append(pathname_list)

path = 'ufi-cropped\\valid\\'
if not os.path.exists(path):
    os.makedirs(path)
for num in range(1, 606):
    if not os.path.exists(path + 's'+ str(num)):
        os.makedirs(path + 's'+ str(num))

for tset in train_set:
    r = random.choice(tset)
    print(r)
    cls = r.split('\\')[2]
    file = r.split('\\')[3]
    shutil.move(r, "ufi-cropped\\valid\\" + cls +"\\" + file)