#!/usr/bin/env python

import pyopencv as cv

img1 = cv.imread('box_in_scene.png',cv.CV_LOAD_IMAGE_GRAYSCALE)
img2 = cv.imread('box.png',cv.CV_LOAD_IMAGE_GRAYSCALE)

surf = cv.SURF()

mask = cv.Mat()

keyp1 = []
descr1 = surf(img1, mask, keyp1)
keyp2 = []
descr2 = surf(img2, mask, keyp2)

N = len(keyp2)
K = 1 # K as in K nearest neighbors
dim = surf.descriptorSize()

m_img1 = cv.asMat(descr1[:].reshape(len(keyp1), dim))
m_img2 = cv.asMat(descr2[:].reshape(N, dim))

flann = cv.Index(m_img1,cv.KDTreeIndexParams(4))

indices = cv.Mat(N, K, cv.CV_32S)
dists = cv.Mat(N, K, cv.CV_32F)

flann.knnSearch(m_img2, indices, dists, K, cv.SearchParams(250))

print indices, dists