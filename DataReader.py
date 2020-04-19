# Helper libraries
import numpy as np
import numpy as np
import glob
import os

class DataReader():

    def find(self, name, path):
        for root, directory, files in os.walk(path):
            if name in files:
                return os.path.join(root, name), root
        return None, None
        
    def find_directory(self, name, path): 
        for root, _, files in os.walk(path):
            
            directory = root.split('/')[-1]

            if name == directory:
                return root
    
    def read_celeb_a(self):

        images = []

        #load labels
        label_file = open('store/list_attr_celeba.txt', 'r')
        labels_lines = label_file.readlines()[2:]

        labels = []
        labels_dict = {}

        for line in labels_lines:

            #find image path
            imagePath = 'store/img_align_celeba/' + line.split(' ', 1)[0]
            images.append(imagePath)

            #extract features
            line = line.strip().split(' ')[1:]

            #labels.append([int(x) for x in line[1:]])
            for label in line:
            
                if (label == '-1' or label == '-1\n'):
                    labels.append(0)      

                elif (label == '1' or label == '1\n'):
                    labels.append(1)

            labels_dict[imagePath] = labels
            labels = []

        return ({ 'train' : images[:int(len(images) * .70)], 'validation': images[int(len(images) * .71) : int(len(images) * .90)], 'test' : images[int(len(images) * 0.91) : int(len(images))] }, labels_dict)