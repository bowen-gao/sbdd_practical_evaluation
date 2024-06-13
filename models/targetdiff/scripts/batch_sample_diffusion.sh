TOTAL_TASKS=100
BATCH_SIZE=20

function cleanup {
  echo "Received termination signal, cleaning up..."
  pkill -P $$
  exit 1
}

trap cleanup EXIT

if [ $# != 6 ]; then
    echo "Error: 6 arguments required."
    echo "Usage: $0 <config_file> <result_path> <node_all> <node_this> <start_idx> <ckpt_path>"
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH=$2
NODE_ALL=$3
NODE_THIS=$4
START_IDX=$5
CKPT_PATH=$6

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH} --ckpt ${CKPT_PATH}
    fi
done
