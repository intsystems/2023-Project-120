IMG="edgenes:v1"
CONT_NAME="edgenes_cont"

docker build . -t ${IMG}

xhost +

docker run \
    -it \
    --rm \
    --hostname user \
    --env="DISPLAY=${DISPLAY}" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --network bridge \
    -p 2222:22 \
    --cap-add=NET_ADMIN \
    --name ${CONT_NAME} \
    -v ${PWD}/checkpoints:/code/checkpoints \
    ${IMG}

echo "[INFO] Finish run container."
