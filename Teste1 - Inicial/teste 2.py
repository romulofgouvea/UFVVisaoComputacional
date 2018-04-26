import  cv2        #disponibiza as funções do opencv 
import numpy as np
import numpy as np
from scipy import misc, ndimage
from skimage import exposure, morphology, img_as_float, filters, util
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

def maiorObjeto(im_L):
    # Seleciona apenas o maior objeto.
    if im_L.max()>1: # Maior valor presente na matriz de rotulos
        max_area = -np.Inf
        max_area_i = -1
        for i in range(1, im_L.max()+1):
            if np.count_nonzero(im_L==i) > max_area:
                max_area = np.count_nonzero(im_L==i)
                max_area_i = i

        im_e = np.zeros(im_L.shape);
        im_e[im_L==max_area_i] = 1.;
        # Matriz de rotulos
        # im_L = measure.label(im_tmp)
        im_L, im_L_size = ndimage.label(im_e)

    #############################
    # Retorna a imagem segmentada
    return im_e

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
    
    retval, threshold = cv2.threshold(median, 0, 255, cv2.THRESH_OTSU)
    
    ee = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    
    #img_close = morphology.binary_closing(threshold, ee)
    #img_open = morphology.binary_opening(img_close, ee)
    
    img_holes = morphology.remove_small_objects(threshold)
    
    img_h = maiorObjeto(img_holes)
    
    plt.imshow(img_holes)
    plt.show()
    
 
def main():
    show_image()
    return 0
 
if __name__ == '__main__':
    main()