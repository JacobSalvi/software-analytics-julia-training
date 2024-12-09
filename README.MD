


install cuda
install openmpi

# Set up the environment
Create a virtual environment:

```bash
python3 -m venv .venv
```

Source the environment:
```bash
sourche .venv/bin/activawe
```

Source the environment on the server:
```bash
eval "$(/home/SA24-G1/miniconda3/bin/conda shell.bash hook)"
```

Install the requirements:
```bash
pip install -r requirements.txt
```

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Train
```bash 
python src/LLM/LLM_train.py --model "1.7b" --sample_run
```
### Train all models
train all models 135m, 360m, 1.7b

```bash
python src/LLM/LLM_train.py --model "all" --sample_run
```

```bash
python src/LLM/LLM_train.py --model "all" --signature --sample_run
```

```bash 
python src/LLM/LLM_train.py --model "all" --baseline --sample_run
```

## Predict

```bash
python src/LLM/LLM_predict.py --prompt "A simple for loop" --model "135m"
```

### Predict all models

```bash
python src/LLM/LLM_predict.py --prompt "A simple for loop" --model "all"
```

```bash 
python src/LLM/LLM_predict.py --prompt "for i in 1:10" --model "all" --signature
```

```bash
python src/LLM/LLM_predict.py --prompt "for i in 1:10" --model "all" --baseline
```