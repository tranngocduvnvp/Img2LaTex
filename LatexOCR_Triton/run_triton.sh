docker run --gpus=all \
 -it --rm --shm-size=256m \
 --rm -p8000:8000 -p8001:8001 \
 -p8002:8002 \
 -v $(pwd)/model_repository:/models \
 -v $(pwd):/workspace \
 nvcr.io/nvidia/tritonserver:24.01-py3 
