from img2tex import cli as img2tex
from PIL import Image
from img2tex.resizeimg4test import resize4test, resize4api


model = img2tex.LatexOCR()

f = r"C:\Users\dutn\Documents\LatexOCR\LatexOCR\img2tex\sample\math1.jpg"


# img = Image.open(f)
def predict(f, scale=1):
    img = resize4test(f, scale=scale)
    math = model(img)
    # print(math)
    return math

print(predict(f))