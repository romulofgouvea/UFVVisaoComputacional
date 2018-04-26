import  cv2        #disponibiza as funções do opencv 
import numpy as np
import numpy as np
from scipy import misc, ndimage
from skimage import exposure, morphology, img_as_float, filters, util
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

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
     
    plt.imshow(res)
    plt.show()
    
 
def main():
    show_image()
    return 0
 
if __name__ == '__main__':
    main()