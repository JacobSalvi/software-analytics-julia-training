


install cuda
install openmpi

```bash
export CUDA_HOME=/opt/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

```bash
export MASTER_ADDR=127.0.0.1   # Localhost for single GPU
export MASTER_PORT=29509       # A free port on your system
export RANK=0                  # Rank is always 0 for a single node
export WORLD_SIZE=1            # Only one process for single GPU
```

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
```

```bash 
python src/LLM/LLM_train.py --model "1.7b" --sample_run
```

```bash
python src/LLM/LLM_predict.py --prompt "A simple for loop" --model "135m"
```