import os
import random

import numpy as np
from keras.preprocessing import image


class SingleViewGen:
    
    def __init__(self, classes_filename, dataset, batch_size, target_size, num_classes, data_folder='view'):
        with open(os.path.join(data_folder, classes_filename), 'r') as cf:
            classes = cf.read().split('\n')[:-1]
        self.paths = []
        for c_idx, c in enumerate(classes):
            curr_path = os.path.join(data_folder, 'classes', c, dataset)
            files = os.listdir(curr_path)
            for f in files:
                for img in os.listdir(os.path.join(curr_path, f)):
                    self.paths.append((c_idx, os.path.join(curr_path, f, img)))
        random.shuffle(self.paths)
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes
        
    def get_num_total_imgs(self):
        return len(self.paths)
    
    def generator(self):
        counter = 0
        while True:
            X = []
            Y = []
            for _ in range(self.batch_size):
                c_idx, img_path = self.paths[counter]
                X.append(image.img_to_array(image.load_img(img_path, target_size=self.target_size)) / 255.)
                y = np.zeros(self.num_classes)
                y[c_idx] = 1
                Y.append(y)
                counter += 1
                if counter >= len(self.paths):
                    counter = 0
            yield np.array(X), np.array(Y)

