import pandas as pd
from img2tex.utils import remove_whitespace

with open("/home/tranngocdu/BentoML/FastAPI/datasets/archive/CROHME/CROHME_math.txt","r") as f:
    s = f.readlines()
    s = [item.strip() for item in s]

images = []
formulas = []

for i, item in enumerate(s):
    images.append("{:07d}.png".format(i))
    formulas.append(remove_whitespace(item))

myFrame = pd.DataFrame({"image":images, "formula":formulas})
myFrame.to_csv("/home/tranngocdu/BentoML/FastAPI/datasets/handwritting.csv", index=False)
