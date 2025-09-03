xhost +

DATA_FOLDER="$PWD/data/cropped_/"
TMP_DATA_FOLDER="/tmp/dataset/$DATA_FOLDER"

mkdir -p $TMP_DATA_FOLDER
rsync -a --info=progress2 -u  $DATA_FOLDER/ $TMP_DATA_FOLDER #--delete

docker  run \
        -it \
        -u $(id -u):$(id -g) \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        --gpus all \
        -v $PWD:$PWD \
        -v $TMP_DATA_FOLDER:$TMP_DATA_FOLDER \
        -p 8888:8888 \
        --name trainer \
        --rm \
        cuda-tensorflow /bin/bash -c "python3 ${PWD}/model/train.py --data_folder $TMP_DATA_FOLDER --output_folder $PWD/out/test"

