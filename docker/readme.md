# Docker environment

## Prerequisites
 - docker-ce
 - nvidia-docker

## Run from local build

### Building
```
sudo sh ./build.sh
```

Optional - push to docker hub
```
sudo docker tag multi-agent-empathy natashajaques/multi-agent-empathy
sudo docker push natashajaques/multi-agent-empathy
```

### Run

```
SEQ_SOC_PATH=/home/natasha/Developer/sequential_social_dilemma_games
RAY_PATH=/home/natasha/Developer/ray
RAY_RESULTS_PATH=/home/natasha/ray_results
sudo docker run --runtime=nvidia -v $SEQ_SOC_PATH:/project -v $RAY_PATH:/ray --rm multi-agent-empathy /bin/bash -c "python setup.py develop && python run_scripts/train_baseline_dqn_actions.py --use_gpu_for_driver --num_gpus=1"
```

## Run from Docker Hub
```
SEQ_SOC_PATH=/home/natasha/Developer/sequential_social_dilemma_games
RAY_PATH=/home/natasha/Developer/ray
sudo docker run --runtime=nvidia -v $SEQ_SOC_PATH:/project -v $RAY_PATH:/ray --rm natashajaques/multi-agent-empathy /bin/bash -c "python setup.py develop && python run_scripts/train_baseline_dqn_actions.py --use_gpu_for_driver --num_gpus=1"
```
