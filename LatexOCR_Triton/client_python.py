import requests
import numpy as np
from PIL import Image
from io import BytesIO
from utils import *
import base64

# Đọc và resize ảnh
img = resize4test(r"/home/bdi/Mammo_FDA/TensorRT/LatexOCR/img2tex/LatexOCR_Triton/img/sample4.jpg", scale=1.0)
image = np.array(img.convert('RGB'))
image = np.expand_dims(np.transpose(test_transform(image=image)['image'][:, :, :1], (2, 0, 1)), axis=0)

# URL của Triton Inference Server
url = "http://localhost:8000/v2/models/pipeline/infer"

# Chuẩn bị dữ liệu để gửi đi
data = {
    "inputs": [
        {
            "name": "input",
            "shape": image.shape,
            "datatype": "FP32",
            "data": image.tolist()
        }
    ],
    "outputs": [
        {
            "name": "respone_latex",
            "datatype": "BYTES"
        }
    ]
}

# Gửi yêu cầu POST đến Triton Inference Server
response = requests.post(url, json=data)

# Kiểm tra xem yêu cầu có thành công hay không
if response.status_code == 200:
    # Lấy kết quả từ phản hồi
    result = response.json()["outputs"][0]["data"]
    
    print("result_latex:", result[0])
    
else:
    print("Error:", response.text)
