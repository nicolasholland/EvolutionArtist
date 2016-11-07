import os
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

# function that compares an image 'img' to a set of Artwork images.
# The idea is, the smaller the distance to the set, the more likely it is, that 'img' can be considered art.
def ArtCritic(img, path):
  critic = 0
  for filename in os.listdir(path):
    # read image
    artwork = cv2.imread(path+'/'+filename)

    # compare
    critic += np.linalg.norm(img-artwork*1.0)
  return critic

def critic2(img, path, filename='violet-black-orange-yellow-on-white-and-red.jpg'):
  artwork = cv2.imread(path+'/'+filename)
  return np.linalg.norm(img-artwork*1.0)




# Test some Stuff

#A = cv2.imread('resizedArt/Black-and-Violet.jpg')
#B = cv2.imread('resizedArt/Bustling-Aquarelle.jpg')

empty = np.zeros((200, 200, 3), np.uint8)
empty[:,:] = (255, 255, 255)

#print np.linalg.norm(A-B)
#print np.linalg.norm(A-empty)

#print ArtCritic(A, 'resizedArt')
print ArtCritic(empty, 'resizedArt')
print critic2(empty, 'resizedArt')
#print ArtCritic(B, 'resizedArt')




