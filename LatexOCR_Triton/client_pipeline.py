from utils import *
import tritonclient.http as httpclient


img = resize4test(r"/home/bdi/Mammo_FDA/TensorRT/LatexOCR/triton/tutorials/Conceptual_Guide/LatexOCR_Triton/img/sample4.jpg", scale=1.0)
image = np.array(img.convert('RGB'))
image = np.expand_dims(np.transpose(test_transform(image=image)['image'][:,:,:1], (2,0,1)), axis=0)

client = httpclient.InferenceServerClient(url="localhost:8000")


image_input = httpclient.InferInput("input", image.shape, "FP32")
image_input.set_data_from_numpy(image)

output_img = httpclient.InferRequestedOutput("respone_latex")
result_latex = client.infer(
        model_name="pipeline", inputs=[image_input], outputs=[output_img]
    ).as_numpy("respone_latex").astype(str)
result_latex = str(result_latex)
print("result_latex:",result_latex)
