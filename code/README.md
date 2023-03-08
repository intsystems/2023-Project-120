### Run basic code

To run a search phase, type the following
```bash
python search.py --decay=decay --save_path=results/arch_decay.json
```

To run a retrain, type the following
```bash
python retrain.py --arc-checkpoint results/arch_decay.json --epochs=7
```