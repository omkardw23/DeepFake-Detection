"""import os
for dirname, _, filenames in os.walk('/content/My Drive/MesoNet/MesoNet/test_videos/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import cv2
plt.style.use('ggplot')

#metadata_file_df = pd.read_json('train_sample_videos/metadata.json').T
#print(metadata_file_df.head())

#stats = metadata_file_df.groupby('label').size()
#stats.plot(figsize = (15,5), kind = 'bar', label = 'Information extracted from JSON Metadata file')

train_dir = '/content/gdrive/My Drive/MesoNet/MesoNet/test_videos/'
# preparing a list of training videos
train_video_files = [train_dir + i for i in os.listdir(train_dir)]
print(train_video_files)

from mtcnn.mtcnn import MTCNN
detector = MTCNN()
v_cap = cv2.VideoCapture(train_video_files[0])
_, frame = v_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12, 8))
plt.imshow(frame, cmap = 'gray')
plt.axis('off')

result = detector.detect_faces(frame)
print(result)
print(len(result))

"""
Adaptive histogram equalization is used OVER Global Histogram Equalization. In this, image is divided into small blocks called 
âtilesâ (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area,
histogram would confine to a small region (unless there is noise).
If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
"""
def img_enhancement(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hist_equalized_image = clahe.apply(image) # Adaptive Histogram equalization 
    # hist_equalized_image = cv2.equalizeHist(image) # Global Histogram Equalization
    return(hist_equalized_image)

"""
The grayscale conversion was done correctly. The problem is in how you are displaying the image.
By default imshow adds it's own colour key to single channel images, to make it easier to view. 
"""

for j in range(len(result)):
    bounding_box = result[j]['box']
    cv2.rectangle(frame,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)
    plt.figure(figsize=(12, 8)) # as cv2.imshow crashes the Jupyter notebook
    plt.imshow(frame)
    plt.axis("off")
    plt.show()
    frame_cropped = frame[bounding_box[1] : bounding_box[1] + bounding_box[3], bounding_box[0] : bounding_box[0] + bounding_box[2]]
    plt.figure(figsize=(12, 8)) # as cv2.imshow crashes the Jupyter notebook
    im = cv2.cvtColor(frame_cropped,cv2.COLOR_RGB2GRAY)
    im = img_enhancement(im)
    plt.imshow(im,cmap = 'gray')
    plt.axis("off")
    plt.show()
    