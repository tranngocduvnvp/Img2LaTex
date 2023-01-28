from fastapi import FastAPI
from fastapi.testclient import TestClient
from .main import app

# app = FastAPI()



client = TestClient(app)


def test_read_main():
    response = client.put("/uploadfile/1",data={"C:\work\FastAPI\datasets\\test_image\ins_2753.png"})
    # assert response.status_code == 200
    assert response.json() =={"filename": "ins_2753.png", "content":"image/png","math":"B.\\ -4<m<4"}

# def test_read_main():
#     response = client.get("/")
#     print(response)
#     assert response.status_code == 200
#     assert response.json() == {"msg": "Hello World"}
