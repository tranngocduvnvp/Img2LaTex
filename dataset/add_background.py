import blend_modes as blm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


img_in = cv2.cvtColor(cv2.imread('/home/tranngocdu/BentoML/FastAPI/datasets/CROHME/0000001.png'), cv2.COLOR_BGR2RGBA).astype(float)/255
img_layer = cv2.cvtColor(cv2.imread('/home/tranngocdu/BentoML/FastAPI/datasets/Background/bg1.png'), cv2.COLOR_BGR2RGBA).astype(float)/255
plt.imshow(img_layer)
img_layer = cv2.resize(img_layer,(img_in.shape[1], img_in.shape[0]))
img_out = blm.darken_only(img_in,img_layer,1.)
img_out = cv2.cvtColor((img_out*255).astype(np.uint8), cv2.COLOR_BGRA2RGB)
print(img_out.shape)
cv2.imwrite("test.png",img_out)
plt.imshow(img_out)
plt.show()