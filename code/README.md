### Run basic code

To run a search phase, type the following, where "checkpoitns/decay" is existing folder
```bash
python search.py --decay=decay --save-folder=checkpoints/decay
```

To run a retrain, type the following, where "checkpoitns/decay" is existing folder with saved "arc.json" in it
```bash
python retrain.py --save-folder=checkpoints/decay --epochs=7
```

To run inference, type the following a directory checkpoints should exist
```bash
python inference.py
```
