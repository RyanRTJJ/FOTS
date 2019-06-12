#!/usr/bin/env python
# coding: utf-8

# # 1 Demo for the COCO-Text data API 

# In this demo, we will learn how to load the COCO-Text data using the python API.
# 
# Let's first import the `coco_text` tool API package. 

# In[1]:


import coco_text
import shutil
import os


# Make sure that you have downloaded the annotation file from the website.
# 
# Once downloaded, you can import the annotations in the following way:

# In[2]:


ct = coco_text.COCO_Text('COCO_Text.json')


# Now, lets use the API. First, the API offers some basic infos of the dataset.

# In[3]:


ct.info()


# ### Select annotations and images based on filter criteria

# Let's retrieve some images. We want to get a list of all image ids from the training set, where the image contains at least one text instance that is legilbe and is machine printed.

# In[4]:


imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible'),('language','english')])
print(imgIds)
print(len(imgIds))


# Let's now go on to the annotations. We want to get a list of all annotation ids from the validation set that are legible, machine printed and have an area between 0 and 200 pixels.

# In[5]:


anns = ct.getAnnIds(imgIds=ct.train, 
                        catIds=[('legibility','legible'),('language', 'english')])
print(anns)
print(len(anns))
                   

# pick one at random
#img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

imgs = ct.loadImgs(imgIds)
IdIterator = 0
while IdIterator < len(imgIds):
  img = imgs[IdIterator]
  imgname = ('/home/Ryan1/FOTS/data/MSCOCO/images/train2014/' + img['file_name'])
  shutil.copy(imgname, '/home/Ryan1/FOTS/data/MSCOCO/PrunedImages')
  
  annIds = ct.getAnnIds(imgIds=img['id'], catIds=[('legibility','legible'),('language','english')])
  anns = ct.loadAnns(annIds)
  print(anns)
  ct.showAnns(anns)
  
  AnnotationsFile = open(("/home/Ryan1/FOTS/data/MSCOCO/annotations" + "/" + img['file_name'][0:-4] + ".txt"), "w")
  annIDIterator = 0
  while annIDIterator < len(annIds):
    AnnotationsFile.write(','.join(str(i) for i in anns[annIDIterator]['polygon']) + ',' + anns[annIDIterator]['utf8_string'] + "\n")
    annIDIterator += 1
  AnnotationsFile.close()
  IdIterator += 1



"""

# # 2 Visualize COCOText Annotations

# In order to visualize the COCO Text annotations, please make sure to download the COCO Images from the MSCOCO website: http://mscoco.org/dataset/#download 

# After downloading the images, specify the path to the MSCOCO image data.

# In[6]:


dataDir='..'
dataType='train2014'


# Lets now import some useful tools to visualize the COCO images and annotations

# In[7]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# Using the API introduced above, lets select an image that has at least one instance of legible text.

# In[8]:


# get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible')])
# pick one at random
img = ct.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


# We can now load the image

# In[9]:


I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
print('/images/%s/%s'%(dataType,img['file_name']))
plt.figure()
plt.imshow(I)


# Lastly, we can load and display the text annotations

# In[10]:


# load and display text annotations
plt.imshow(I)
annIds = ct.getAnnIds(imgIds=img['id'])
anns = ct.loadAnns(annIds)
ct.showAnns(anns)


# # 3 Demo for the COCO-Text evaluation API

# In this demo we will learn how to use the COCO-Text evaluation API to evaluate text detection and recognition results.
# 
# First, let's import the `coco_text_evaluation` API.  

# In[11]:


import coco_evaluation


# Next, we have to load our recognition results. For this we can use the `loadRes()` function from the `coco_text` tool.
# 
# The results have to be saved in the format explained on the website. The '`our_results.json`' file gives an example. Generally, the detections are saved in a json file and form a list of dictionaries like the following:

#     [{"image_id": int,
#       "bbox": [left, top, width, height],
#       "utf8_string": string"},
#       {}...]

# Then, we can load the results like this:

# In[12]:


our_results = ct.loadRes('our_results.json')


# If the results file contains annotations for images not in the current version of COCO-Text, the loader will notify
# that some images are skipped and then ignore the respective annotations. This happens for example, if results for the test set are included in the same file.

# ### Detection results

# Once the resutls are loaded, the evalution tool allows to compute the successful detections with the '`getDetections()`' function. The `detection_threshold` parameter defines how closely the bounding boxes need to overlap. The default value is an Intersection over Union (IoU) score of 0.5. 

# In[13]:


our_detections = coco_evaluation.getDetections(ct, our_results, detection_threshold = 0.5)


# The detection results comprise three lists: True Positives, False Positives and False Negatives.

# In[14]:


print('True positives have a ground truth id and an evaluation id: ', our_detections['true_positives'][0])
print('False positives only have an evaluation id: ', our_detections['false_positives'][0])
print('True negatives only have a ground truth id: ', our_detections['false_negatives'][0])


# ### End-to-end results

# Let's look into the transcription performance now. For that we ue the '`evaluateTranscription()`' function. And provide our results and detections.

# In[15]:


our_endToEnd_results = coco_evaluation.evaluateEndToEnd(ct, our_results, detection_threshold = 0.5)


# Now we are ready to see the results. For that we can use the '`printDetailedResults()`' function. The last line can be used to create a table as shown in the paper.

# In[16]:


coco_evaluation.printDetailedResults(ct,our_detections,our_endToEnd_results,'our approach')

"""