
REPO_ROOT=$(git rev-parse --show-toplevel)

docker  run \
    -u $(id -u):$(id -g) \
    --gpus all \
    -v $PWD:$PWD \
    cuda-tensorflow /bin/bash -c "python3 $REPO_ROOT/run_inference_step1.py \
        --model $REPO_ROOT/models/m330_07/model.onnx \
        --input_folder $REPO_ROOT/<INPUT FOLDER> \
        --output_folder $REPO_ROOT/step1_output/<OUTPUT FOLDER> "
