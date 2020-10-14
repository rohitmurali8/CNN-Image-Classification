import cv2
import numpy as np
import os
import pickle
def preprocess(fold_path, data_filename, label_filename, target_image_size = 150):
    img_size = target_image_size
    training_dataset = []
    categories = ["dogs", "cats"]
    for category in categories:
        def preprocessing_data(image_path):
            for Image in os.listdir(image_path):
                try:
                    img = cv2.imread(os.path.join(image_path, Image))
                    img_resize = cv2.resize(img, (img_size, img_size))
                    training_dataset.append([img_resize, categories.index(category)])
                except:
                    pass
        image_path=os.path.join(fold_path, category)
        print("start preprocessing images ... ", category)
        preprocessing_data(image_path)
        print("read in and resized the images of", category)
    # Randomize the traning dataset
    np.random.shuffle(training_dataset)
    X = [] # features
    y = [] # label
    for features, label in training_dataset:
        X.append(features)
        y.append(label)
        
    X = np.array(X).reshape(-1, img_size, img_size, 3)
    pickle_X = open(data_filename, "wb")
    pickle.dump(X, pickle_X)
    pickle_X.close()
    print("Saved the image data to ", data_filename)
    pickle_y=open(label_filename, "wb")
    pickle.dump(y, pickle_y)
    pickle_y.close()
    print("Saved the label data to ", label_filename)
    
fold_path = "./project1/train" # the folder path to the train data set
data_filename = "X_train.pickle"
label_filename = "y_train.pickle"
print("start preprocessing the train data set ....")
preprocess(fold_path, data_filename, label_filename, target_image_size = 150)
print("completed preprocessing the train data set.")

fold_path = "./project1/validation" # the folder path to the validation data set
data_filename = "X_validation.pickle"
label_filename = "y_validation.pickle"
print("start preprocessing the validation data set ....")
preprocess(fold_path, data_filename, label_filename, target_image_size = 150)
print("completed preprocessing the validation data set.")
