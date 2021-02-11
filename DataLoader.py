import pandas as pd
import numpy as np
import os
import pickle


#Read in the labels from the text file in csv format#
train_df = pd.read_csv("../input/vinbigdata-chest-xray-abnormalities-detection/train.csv")

print(train_df.head())

#Initialize a list for the unique IDs in all the findings#
image_ids = []

#Using the index method to return the number of rows in the label file, in other#
#words it is the total number of findings in all of the images from all Radiologists)#
f = train_df.index

print(len(f))

#To group by image, we first need a list of the unique images names from the findings#
#We can do this easily using the .unique() method on the image_id column of the labels#
#file#
image_ids = list(train_df['image_id'].unique())
#We also need a list of image IDs for each finding#
findings = list(train_df['image_id'])

print(findings[59000])

print(image_ids[12])

#Initializing lists labels_img_rad will be used to create final_labels and boxes_img_rad will do the#
#same for the associated final_boxes#
labels_img_rad = []
boxes_img_rad = []
final_labels = []
final_boxes = []

#The structure for final_labels and final_boxes is a list for each image containing three lists for the#
#findings of the three radiologists. In final_labels each list is number corresponding to the category of#
#that radiologist's findings, while in final_boxes each list is a list of 4 numbers corresponding to#
#the bounding box for that radiologist's findings#
for img in image_ids:
   labels_img_rad.append([[],img])
   boxes_img_rad.append([[],img])
   final_labels.append([[],[],[]])
   final_boxes.append([[],[],[]])
i = 0
for name in findings:
   idx = image_ids.index(name)
   split = list(train_df.iloc[i])
   labels_img_rad[idx][0].append({split[3]:int(split[2])})
   boxes_img_rad[idx][0].append({split[3]:split[4:]})
   #print(i)
   i += 1

#This loop uses label_img_rad to fill final_labels with the sorted labels#
i = 0
for img in labels_img_rad:
   rad1 = []
   rad1f = []
   rad2 = []
   rad2f = []
   rad3 = []
   rad3f = []

   for dic in img[0]:

      if dic.keys() in rad1 or rad1 == []:
         rad1.append(dic.keys())
         rad1f.append(list(dic.values())[0])

      elif dic.keys() in rad2 or rad2 == []:
         rad2.append(dic.keys())
         rad2f.append(list(dic.values())[0])

      elif dic.keys() in rad3 or rad3 == []:
         rad3.append(dic.keys())
         rad3f.append(list(dic.values())[0])

   final_labels[i][0] = rad1f
   final_labels[i][1] = rad2f
   final_labels[i][2] = rad3f
   i += 1

    
#This loop uses boxes_img_rad to fill final_boxes with the sorted bounding boxes#
i = 0
for img in boxes_img_rad:
   rad1 = []
   rad1f = []
   rad2 = []
   rad2f = []
   rad3 = []
   rad3f = []
   for dic in img[0]:
      if dic.keys() in rad1 or rad1 == []:
         rad1.append(dic.keys())
         rad1f.append(list(dic.values())[0])
      elif dic.keys() in rad2 or rad2 == []:
         rad2.append(dic.keys())
         rad2f.append(list(dic.values())[0])
      elif dic.keys() in rad3 or rad3 == []:
         rad3.append(dic.keys())
         rad3f.append(list(dic.values())[0])

   final_boxes[i][0] = rad1f
   final_boxes[i][1] = rad2f
   final_boxes[i][2] = rad3f


   i += 1
print(final_boxes)
print(labels_img_rad[14144])
print(boxes_img_rad[13])

#Finally use the pickle module to save output#
pickle.dump( final_boxes, open( "./final_boxes.pickle", "wb" ) )
pickle.dump( final_labels, open( "./final_labels.pickle", "wb" ) )
pickle.dump( image_ids, open( "./image_ids.pickle", "wb" ) )
