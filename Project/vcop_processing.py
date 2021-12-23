import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def gray_scale_equalizing(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalize_image = cv2.equalizeHist(image)
    stacked_img = np.stack((equalize_image,) * 3, axis=-1)
    return stacked_img

def border_extraction(img):
    kernel = np.ones((3,3), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    img_border = img - img_erosion
    return img_border

def main_001():
    path_dataset = './dataset'

    classes_list = []
    
    for entry in os.scandir(path_dataset):
        if entry.is_dir():
            classes_list.append(entry.name)

    for class_index, class_name in enumerate(classes_list):
        print(f'Processing Data of Class: {class_name}')
        for file in os.listdir(path_dataset + '/' + class_name):
            if file.endswith('_color.jpg'):
                filename = file.replace('_color.jpg', '')
                img_color = cv2.imread(path_dataset + '/' + class_name + '/' + filename + '_color.jpg')
                try:
                    # cv2.imwrite(path_dataset + '/' + class_name + '/' + filename + '_gray.jpg', gray_scale_equalizing(img_color))
                    cv2.imwrite(path_dataset + '/' + class_name + '/' + filename + '_border.jpg', border_extraction(img_color))
                except:
                    continue

def main_002():
    path_dataset = './dataset'
    classes_list = []
    df = pd.DataFrame(columns = ['File', 'Class'])
    
    for entry in os.scandir(path_dataset):
        if entry.is_dir():
            classes_list.append(entry.name)
            
    for class_index, class_name in enumerate(classes_list):
        print(f'Processing Data of Class: {class_name}')
        for file in os.listdir(path_dataset + '/' + class_name):
            if file.endswith('_color.jpg'):
                filename = file.replace('_color.jpg', '')
                
                df = df.append({'File' : filename, 'Class' : class_name}, ignore_index = True)
                
    df = df.groupby('File')['Class'].apply(list).reset_index(name='Classes')
    
    for index, row in df.iterrows():
        filename = row['File']
        classes = row['Classes']
        binary_classes = np.zeros(len(classes_list))
        
        for indx, clas in enumerate(classes_list):
            if clas in classes:
                binary_classes[indx] = 1
        
        df.loc[index, 'Classes'] = binary_classes.tolist()
    df.to_csv('out.csv')
    print(classes_list)
    
def main_003():
    path_dataset = './dataset'

    classes_list = []
    
    for entry in os.scandir(path_dataset):
        if entry.is_dir():
            classes_list.append(entry.name)

    for class_index, class_name in enumerate(classes_list):
        print(f'Processing Data of Class: {class_name}')
        count = 0
        for file in os.listdir(path_dataset + '/' + class_name):
            if file.endswith('_color.jpg'):
                count +=1
        print(count)

main_003()