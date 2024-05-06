### Functionality description

The main three functions of the code are:

- __Search for architectures__: search architectures with one of the following methodologies:
    - __Optimal__ - search architectures via vanilla DARTS.
    - __Edges__ - search architectures with exploitation of a regularizer with fixed parameter $\lambda$.
    - __Random__ - randomly change specified amount of edges in the basic architecture.
    - __Hypernet__ - novel method that exploits hypernet and samples lambda on every iteration of optimization.
- __Retrain architectures__: retrain weights of specified architecture starting from random initialisation.
- __Inference models as ensemble__: choose models and aggregate their answers. Compute their performance as ensemble.

These functions are controlled via config files, located in `./configs`. Code saves architectures and models after searching and retraining in specified directory.

### Launch code

- For searching
Edit ./configs/search.yaml for your purposes
    - `chmod +x run_search.sh`
    - `./run_search.sh`
- For retrain
Edit ./configs/retrain.yaml for your purposes
    - `chmod +x run_retrain.sh`
    - `./run_retrain.sh`
- For inference
Edit ./configs/inference.yaml for your purposes
    - `chmod +x run_inference.sh`
    - `./run_inference.sh`

