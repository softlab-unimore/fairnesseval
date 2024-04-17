# Fairness Evaluation and Testing Repository

Automated decision-making systems can potentially introduce biases, raising ethical concerns. This has led to the
development of numerous bias mitigation techniques.
However, the selection of a fairness-aware model for a specific dataset often involves a process of trial and error, as
it is not always feasible to predict in advance whether the mitigation measures provided by the model will meet the
user's requirements, or what impact these measures will have on other model metrics such as accuracy and run time.

Existing fairness toolkits lack a comprehensive benchmarking framework. To bridge this gap, we present FairnessEval, a
framework specifically designed to evaluate fairness in Machine Learning models. FairnessEval streamlines dataset preparation,
fairness evaluation, and result presentation, while also offering customization options.
In this demonstration, we highlight the functionality of FairnessEval in the selection and validation of fairness-aware models.
We compare various approaches and simulate deployment scenarios to showcase FairnessEval effectiveness.


## fairnesseval DEMO
[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/DEMO%20fairnesseval.ipynb)
you can find a demo notebook with working examples.

You can interact with the notebook and run the library with your experiments.

## fairnesseval API Quick Start

[Here](https://github.com/softlab-unimore/fairnesseval/blob/main/notebooks/fairnesseval%20Quick%20Start.ipynb)
you can find a quick start guide to the fairnesseval API with working examples.


[//]: # ()

[//]: # (# Scalable Fairlearn)

[//]: # ()

[//]: # ()

[//]: # (## Example Runs)

[//]: # ()

[//]: # (#### Synth)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth hybrids --eps=0.05 -n=10000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.016 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth hybrids --eps=0.05 -n=1000000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.016 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # (##### Unmitigated)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth unmitigated -n=10000 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3)

[//]: # (```)

[//]: # ()

[//]: # (##### Fairlearn)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth fairlearn --eps=0.05 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 -n=10000)

[//]: # (```)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py synth fairlearn --eps=0.05 -f=3 -t=0.5 -t0=0.3 -t1=0.6 -v=1 --test_ratio=0.3 -n=1000000)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # (#### Adult)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult unmitigated)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult fairlearn --eps=0.05)

[//]: # (```)

[//]: # (```)

[//]: # (time stdbuf -oL python run.py adult hybrids --eps=0.05 --sample_seeds=0,1,2,3,4,5,6,7,8,9 --train_fractions=0.001,0.004,0.016,0.063,0.251,1 --grid-fraction=0.5)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # ()

[//]: # ()

[//]: # (## TODOs)

[//]: # ()

[//]: # (### Complete Hybrid Method)

[//]: # (* Single hybrid method that gets the best of all hybrid methods we have)

[//]: # (* Show that it works on both train and test data)

[//]: # ()

[//]: # (### Scaling experiments)

[//]: # (* Show running time savings when dataset is very large &#40;use synthetic data&#41;)

[//]: # (* Also try logistic regression on large image dataset)

[//]: # ()

[//]: # (### Multiple datasets)

[//]: # (* Show it works on three datasets)

[//]: # (* Try logistic regression on large image dataset)

[//]: # ()

[//]: # (### Increasing number of attributes)

[//]: # (* Decide if we can do that experiment...)

[//]: # ()

[//]: # (### Other things)

[//]: # (* How to subsample for the scalability plot to ensure + and - points are treated equally &#40;stratified data sampling?&#41;)
