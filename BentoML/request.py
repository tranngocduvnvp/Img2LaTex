import requests
from PIL import Image
import numpy as np

img = Image.open("C:\Work\FastAPI\datasets\\test_image\ins_2753.png")

text = requests.post(
   "http://localhost:3000/predict",
   headers={"content-type": "application/json"},
   data={"image":img}
).text
print(text)