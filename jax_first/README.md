
Install:
```
python -m venv jax_first.venv
source jax_first.venv/bin/activate
python -m pip install -r requirements.txt
```


```
nsys profile --force-overwrite true -o /tmp/attention.nsys-rep --gpu-metrics-device all python attention.py
# Use preset from GPT-3 and adjust batch size and sequence length.
nsys profile --force-overwrite true -o /tmp/attention.nsys-rep --gpu-metrics-device all python attention.py --num_sentences 16 --sequence_length 1024 --preset=0
```
