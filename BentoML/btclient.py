from bentoml.client import Client
from PIL import Image

img = Image.open("/home/tranngocdu/BentoML/FastAPI/datasets/test_image/ins_2753.png")

client = Client.from_url("http://localhost:3000")
client.inference(img)