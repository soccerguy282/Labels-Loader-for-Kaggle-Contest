import pandas as pd
import numpy as np
import os
#Put your path here#
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


labels_img_rad = []
boxes_img_rad = []
final_labels = []
final_boxes = []


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
   print(i)
   i += 1


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
         rad1f.append(list(dic.values()))
      elif dic.keys() in rad2 or rad2 == []:
         rad2.append(dic.keys())
         rad2f.append(list(dic.values()))
      elif dic.keys() in rad3 or rad3 == []:
         rad3.append(dic.keys())
         rad3f.append(list(dic.values()))

   final_boxes[i][0] = rad1f
   final_boxes[i][1] = rad2f
   final_boxes[i][2] = rad3f

   i += 1

print(final_labels)
print(labels_img_rad[14144])

print(boxes_img_rad[13])
print(final_labels[14988])
print(final_boxes[14988])
