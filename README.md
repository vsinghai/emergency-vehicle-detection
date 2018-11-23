# Northrop Grumman Coding Challenge
Image detection program submission for the **Northrop Grumman Coding Challenge**.
 ## Contents
- [Image]
- [Python]
- [Confidence]
 **Back End**
## Image
* ** Take in Array of Strings to image location and parse 
## Python
* Parsed using Python, installation links below:  
  * [imutils](https://pypi.org/project/imutils/)
  * [scikit-image](http://scikit-image.org/docs/dev/install.html)
  * [numpy](http://www.numpy.org/)
  * [argparse](https://docs.python.org/3/library/argparse.html)
  * [opencv](https://pypi.org/project/opencv-python/)
 ## Strategy
### 1. Creating Nueral Network
* **Neural network to detect vehicles within an image **
  * Created a neural network that takes in an image and boxes vehicles within it, we than 
  * took those points of interest and pushed it to the python scripts for further filtering.
### 2. Filtering The Image:
* **Template Matching**
* **Bright Light Detection**
* **Color Detection **
* Each time an image is passed, we filtered the image using filters we made and pushed 
  * points of interest into an array for future use.
### 3. Confidence Interval
* ** Took clusters of points of interests and determined if they are close enough to quantitatively
     say there is an emergency vehicle at this location