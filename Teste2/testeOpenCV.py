import  cv2        #disponibiza as funções do opencv 
import numpy as np
from scipy import misc, ndimage
from skimage import exposure, morphology, img_as_float, filters, util, measure
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

def maiorObjeto(grayscaled,threshold):
    
    im_r,tmp = ndimage.label(threshold)
    
    props = measure.regionprops(im_r,grayscaled)
    objMaior = 0
    img_m = 0
    for i in range(0, im_r.max()):
        im_t = np.zeros(im_r.shape)
        
        im_t [im_r == i+1] = 1
        
        
        if props[i].area > objMaior:
            objMaior = props[i].area
            img_m = im_t
            
    return img_m

def filtroRGB(src,r,g,b):
    if r == 0:
        src[:,:,2] = 0    #elimina o vermelho
    if g == 0:
        src[:,:,1] = 0    #elimina o verde
    if b == 0:
        src[:,:,0] = 0    #elimina o azul
 
def show_image():   
    img = cv2.imread('planta2.png')
    
    filtroRGB(img,0,1,0)
    verde_inferior = np.array([0,127,0])
    verde_superior = np.array([255,255,255])
    mascara = cv2.inRange(img, verde_inferior, verde_superior)
    res = cv2.bitwise_and(img,img, mask= mascara)          

    grayscaled = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    
    median = cv2.medianBlur(grayscaled,3)
    
    retval, threshold = cv2.threshold(median, 10, 255, cv2.THRESH_OTSU)
    
    elemEstr = np.array([[0,1,0],
                         [1,1,1],
                         [0,1,0]])
    
    kernel3 = np.ones((3,3),np.uint8)
    kernel5 = np.ones((5,5),np.uint8)
    
    #np.ones((3,3))
    selem = morphology.disk(3)
    
    #img_close = morphology.binary_closing(grayscaled, elemEstr)
    #img_top_hat = morphology.white_tophat(img_close,selem)
    
    #img_open = morphology.binary_opening(img_top_hat, elemEstr)
    
    #img_holes = morphology.remove_small_objects(img_open)
    
    
    
    #fechamento
    #transformacion = cv2.morphologyEx(grayscaled,cv2.MORPH_CLOSE,kernel)
    
    #tophat
    #topHat = cv2.morphologyEx(grayscaled,cv2.MORPH_TOPHAT,kernel5)
    
    #abertura
    abertura = cv2.morphologyEx(grayscaled,cv2.MORPH_OPEN,kernel3)
    
    #dilatação
    dilatacao = cv2.dilate(abertura,kernel5,iterations = 1)
    
    #erosao
    erosao = cv2.erode(dilatacao,kernel3,iterations = 1)
    
    #linha
    #img_tra = erosao
    #linha = cv2.dilate(img_tra,kernel3,iterations = 1) - cv2.erode(img_tra,kernel3,iterations = 1)
    
    plt.figure()
    img_print = erosao
    img_maior = maiorObjeto(grayscaled,img_print)
    #plt.imshow(maiorObjeto(grayscaled,img_open))
    misc.imsave('teste.tif',img_print)
    misc.imsave('teste_maior.tif',img_maior)
    
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_print, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_maior, cmap='gray')

    plt.show()
    
 
def main():
    show_image()
    return 0
 
if __name__ == '__main__':
    main()