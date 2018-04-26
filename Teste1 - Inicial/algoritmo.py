# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc, ndimage
from skimage import exposure, morphology, img_as_float, filters, util
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
#import cv2

from skimage.morphology import watershed
from skimage.feature import peak_local_max

def funcWatershed(image):
    x, y = np.indices((80, 80))
    #x1, y1, x2, y2 = 28, 28, 44, 52
    #r1, r2 = 16, 20
    #mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    #mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    #image = np.logical_or(mask_circle1, mask_circle2)
    
    
    
    
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background    
    distance = ndimage.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=image)
    markers = ndimage.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)
    
    num_l, num_c = image.shape
    mascara = np.zeros([num_l, num_c], dtype=float) 
    
    imN = distance*mascara
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(imN, cmap=plt.cm.spectral, interpolation='nearest')
    ax[2].set_title('Separated objects')
    
    for a in ax:
        a.set_axis_off()
    
    fig.tight_layout()
    plt.show()

def prova(nomeImagem):
    
    img = np.array(Image.open(urlImageJPG).convert('L'))
    
    imagem= img_as_float(img)
    
    # Filtro gaussiano com sigma = 5
    #gauss_5x5 = ndimage.filters.gaussian_filter(imagem,sigma=5, mode='constant', cval=0)
    mediana_5x5 = ndimage.filters.median_filter(imagem, size=5, mode='constant', cval=0) 
    
    #median = cv2.medianBlur(imagem,5)
    
    img_eq = exposure.equalize_hist(imagem)
        
    img_otsu = filters.threshold_otsu(mediana_5x5)
    img_otsu = mediana_5x5 < img_otsu
    
    
    ee = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    
    
    img_close = morphology.binary_closing(img_otsu, ee)
    img_open = morphology.binary_opening(img_close, ee)
    
    im = morphology.binary_erosion(img_otsu, ee)
    img_holes = ndimage.binary_fill_holes(im, ee)
    
    #util.invert(img_holes)
    #funcWatershed(img_holes)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.imshow(img_eq, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_otsu, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.imshow(img_holes, cmap='gray')
    
    misc.imsave('img.tif',img)
    misc.imsave('img_eq.tif',img_eq)
    misc.imsave('img_otsu.tif',img_otsu)
    misc.imsave('img_open.tif',img_holes)
    plt.show()   

#urlImage = 'D:/ARQUIVOS/MATERIAL ACADEMICO/UFV/6 - Periodo/SIN 393 - INTRODUÇÃO À VISÃO COMPUTACIONAL/sin393-p1-iranildo_3903-Romulo_3971/imagem10.tif'
#urlImageJPG = 'D:/ARQUIVOS/MATERIAL ACADEMICO/UFV/6 - Periodo/SIN 393 - INTRODUÇÃO À VISÃO COMPUTACIONAL/sin393-p1-iranildo_3903-Romulo_3971/imagem10.jpg'
urlImageJPG = 'D:/ARQUIVOS/MATERIAL ACADEMICO/UFV/6 - Periodo/SIN 393 - INTRODUÇÃO À VISÃO COMPUTACIONAL/sin393-p1-iranildo_3903-Romulo_3971/planta2.png'
prova(urlImageJPG)
