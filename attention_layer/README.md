Instructions:

Install:
```
cd attention_layer
python -m venv attention_layer.venv
source attention_layer.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install transformers # TODO: Debug why this is a separate step.
python -m pip install -r requirements.txt
```

Run it:
```
nsys profile --force-overwrite true -o /tmp/attention_layer.nsys-rep --gpu-metrics-device all python attention_layer.py --num_sentences=1 --sequence_length=512
```
