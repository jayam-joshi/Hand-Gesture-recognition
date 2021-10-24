# Hand-Gesture-recognition
Predicting the digit lying between 1 and 6 depicted by hand gesture using convolutional neural network and computer vision.## Libraries and tools
## Tools and languages
<img src="https://github.com/github/explore/raw/main/topics/python/python.png" width="50" height="50" />        <img src="https://github.com/github/explore/raw/main/topics/tensorflow/tensorflow.png" width="50" height="50" />       <img src="https://github.com/github/explore/raw/main/topics/opencv/opencv.png" width="50" height="50" />

## CNN architecture used
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential_1 (Sequential)    (None, 64, 64, 3)         0
_________________________________________________________________
sequential (Sequential)      (None, 64, 64, 3)         0
_________________________________________________________________
conv2d (Conv2D)              (None, 62, 62, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 32)        9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 12, 32)        9248
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 32)          0
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0
_________________________________________________________________
dense (Dense)                (None, 128)               147584
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 903
=================================================================
Total params: 167,879
Trainable params: 167,879
Non-trainable params: 0
```
## OpenCV
To obtain a binary image frame of the hand the following techniques are used
### Background Subtraction
To seperate hand from background
### Thresholding
To detect the hand region from the difference image, difference image is thresholded, so that only the hand region becomes visible and pixels of all unwanted regions are set to zero.
![Capture](https://user-images.githubusercontent.com/82452505/138595329-dad9b00f-3599-4408-b174-29a6de13fa7f.PNG)
###contribution
There is scope to improve frame rate at which hand gesture is detected so that model can be used efficiently in real time applications.




