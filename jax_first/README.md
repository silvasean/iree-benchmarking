
Install:
```
python -m venv jax_first.venv
source jax_first.venv/bin/activate
python -m pip install -r requirements.txt
```


```
nsys profile --force-overwrite true -o /tmp/attention.nsys-rep --gpu-metrics-device all python attention.py
```
