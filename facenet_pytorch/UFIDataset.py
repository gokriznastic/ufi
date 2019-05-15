import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm

class UFIDataset(datasets.ImageFolder):
    '''
    '''
    def __init__(self, dir, pairs_path, transform=None):

        super(UFIDataset, self).__init__(dir,transform)

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_ufi_paths(dir)

    def read_ufi_pairs(self,pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines():
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_ufi_paths(self,lfw_dir,file_ext="pgm"):

        pairs = self.read_ufi_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for i in tqdm(range(len(pairs))):
        #for pair in pairs:
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(lfw_dir,'train', pair[0], pair[1]+'.'+file_ext)
                path1 = os.path.join(lfw_dir,'test', pair[0], pair[2]+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(lfw_dir,'test', pair[0], pair[1]+'.'+file_ext)
                path1 = os.path.join(lfw_dir,'test', pair[2], pair[3]+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list.append((path0,path1,issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
            # break
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        '''

        Args:
            index: Index of the triplet or the matches - not of a single image

        Returns:

        '''

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1,path_2,issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame


    def __len__(self):
        return len(self.validation_images)
