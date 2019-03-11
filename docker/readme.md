# Docker environment

## Prerequisites
 - docker-ce
 - nvidia-docker

## Building
```
sudo sh ./build.sh
```

## Run

```
SEQ_SOC_PATH=/home/natasha/Developer/sequential_social_dilemma_games
RAY_PATH=/home/natasha/Developer/ray
sudo docker run --runtime=nvidia -v $SEQ_SOC_PATH:/project -v $RAY_PATH:/ray --rm multi-agent-empathy /bin/bash -c "/bin/bash python setup.py develop && python run_scripts/train_baseline_dqn_actions.py --use_gpu_for_driver --num_gpus=1"
```
