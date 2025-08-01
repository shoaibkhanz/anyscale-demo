import marimo

__generated_with = "0.14.15"
app = marimo.App(
    width="columns",
    layout_file="layouts/presentation_notebook.slides.json",
)

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    ## Agenda

    <ul>
      <li><span style="font-size:1.25em;"><b>AI demand</b></span></li>
      <li><span style="font-size:1.25em;"><b>Python Computing Challenges</b></span></li>
      <li><span style="font-size:1.25em;"><b>Ray Primitives</b></span></li>
      <li><span style="font-size:1.25em;"><b>Ray Libraries (Ray Data/Train/Tune and Serve)</b></span></li>
      <li><span style="font-size:1.25em;"><b>Ray on K8s using KubeRay</b></span></li>
      <li><span style="font-size:1.25em;"><b>Anyscale Console</b></span></li>
    </ul>
    """
    )
    return


@app.cell(hide_code=True)
def _():

    mo.vstack([mo.md("## AI Demand"),
    mo.hstack([
        mo.image("marimo_notebooks/resources/ai-workloads.png", height=500, width=700),
        mo.md(
            "<div style='height:80px;'></div>"
            "- **AI** is no longer niche; it’s the core driver of **compute** and **infrastructure demand**.\n"
            "- We are seeing a **3.5× surge** in **AI workloads** projected by 2030.\n"
            "- To **stay competitive**, organisations will need intelligent, adaptive strategies to manage this **growth** and **scale** efficiently.\n"
            "- The article also highlights an expected investment of **$3 trillion to $8 trillion** by 2030.\n"
            "\n"
            "[Complete Mckinsey Article](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/the-cost-of-compute-a-7-trillion-dollar-race-to-scale-data-centers)"
        )
    ])
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        """
    ## AI Trends
    ### AI Data Processing

    - **Tabular** &rarr; **Multimodal** (audio, video, text, etc.)
    - **Data Processing** &rarr; **Inference + Data Processing**
    - **CPUs** &rarr; **CPUs + GPUs** (hybrid)

    ### Agentic Workflows

    - Agentic AI systems are dynamic and complex.
    - They comprise compositions of multiple models, tools, prompts, & stateful logic.
    - Orchestration, evaluation, and observability become a core part of serving them.

    ### Post Training

    - Post training is where enterprises realise value, with fine-tuning, RLHF, etc.

    [Source: Anyscale Infra Summit 2025](https://www.youtube.com/watch?v=cIPsdmiQAog)
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    ## Building Scalable and Performant Solutions

    {mo.image("marimo_notebooks/resources/ray.png")}

    ## What is RAY ?
    Ray is an open-source framework to build and scale ML and Python applications. It consists of easy to use APIs that allows the developer to run their workloads in heterogenous compute (CPUs and GPUs), supports any data type and model architecture, scaling to thousands of GPUs while keeping a stable utilisation across all your compute.

    > Anyscale is a commercial provider that offers a managed platform built on top of Ray with further optimisations and new robust features such as [Ray Turbo](https://www.anyscale.com/product/platform/rayturbo) and others. It abstracts away the operational complexity of deploying and managing Ray clusters, enabling teams to focus on developing scalable applications without worrying about infrastructure.


    {mo.image("marimo_notebooks/resources/anyscale_features.png")}
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        rf"""
    ## Ray Ecosystem 
    {mo.image("marimo_notebooks/resources/ray-libraries.png",height=400, width=550)}

    **Ray Core** is the low-level API that provides the building blocks for distributed computing in Python. It enables developers to define scalable classes and functions for parallel and fault-tolerant workloads.

    Built on top of Ray Core are higher-level libraries that simplify key stages of the AI/ML lifecycle:

    - **Ray Data**: Distributed data loading and preprocessing for tabular, image, and text data. Integrates with Ray Train and Ray Serve.

    - **Ray Train**: Scales training for frameworks like [PyTorch](https://pytorch.org/), [XGBoost](https://xgboost.readthedocs.io/en/stable/) and many others. Works seamlessly with Ray Data and Ray Tune.

    - **Ray Tune**: Hyperparameter tuning at scale with advanced search algorithms like ASHA and Bayesian optimization and integrations with ML optimisation libraries like [Optuna](https://docs.ray.io/en/latest/tune/examples/optuna_example.html) and others.


    - **Ray Serve**: Model serving for real-time inference with support for request batching and multi-model composition.

    Together, these components enable scalable, production-grade AI pipelines from data to deployment using a unified API surface built on Ray Core. Anyscale is the commercial and further optimised offering of Ray from the Ray developers. For more information see [Ray Docs](https://docs.ray.io/en/latest/index.html) and [Anyscale Docs](https://docs.anyscale.com/)
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        f"""
    ## Scaling Python Applications

    {mo.hstack([
        mo.image("marimo_notebooks/resources/concurrent-burgers.png", height=400, width=800),
        mo.image("marimo_notebooks/resources/parallel-burgers-01.png", height=400, width=800),
    ])}
    """
    )
    return


@app.cell
def _():
    # lets import thread and process executor
    import time
    from concurrent.futures.thread import ThreadPoolExecutor
    from concurrent.futures import as_completed

    # now we create a simple sum_of_squares function
    def sum_of_squares(n):
        start = time.time()
        total = sum(i*i for i in range(n))
        total_time = time.time() - start
        return total, total_time

    # lets test the function with the following inputs
    inputs = [ 4 ,10_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000, 70_000_000]
    return ThreadPoolExecutor, as_completed, inputs, sum_of_squares, time


@app.cell
def _(inputs, sum_of_squares, time):
    # lets first run them sequentially
    start_time = time.time()
    for input in inputs:
        result, execution_time = sum_of_squares(input)
        print(f"input: {input} -> {result} -> time taken {execution_time:.4f}s")
    print(f"total time taken {(time.time() - start_time):.4f}s")

    return


@app.cell
def _(ThreadPoolExecutor, as_completed, inputs, sum_of_squares, time):
    # f(input1) f(input2) f(input3) ....
    def thread_pool(func,inputs):
        start_time = time.time()
        with ThreadPoolExecutor() as thread_executor:
            futures = {thread_executor.submit(func,input) for input in inputs}
            for future in as_completed(futures):
                result, execution_time = future.result()
                print(f"{result} -> time taken {execution_time:.4f}s")
        print(f"total time taken {(time.time() - start_time):.4f}s")

    thread_pool(sum_of_squares,inputs)

    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    > *NOTE: ProcessPool is breaks in a notebook environment, since it needs to spawn another Python environment, but essentially it spins multiple python processes to execute the function*

    ```python

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    def process_pool(func, inputs):
        import time

        start_time = time.time()
        with ProcessPoolExecutor() as process_executor:
            futures = {process_executor.submit(func, input) for input in inputs}

            for future in as_completed(futures):
                result, execution_time = future.result()
                print(f"{result} -> time taken {execution_time:.4f}s")
        print(f"total time taken {(time.time() - start_time):.4f}s")
    # now we create a simple sum_of_squares function
    def sum_of_squares(n):
        start = time.time()
        total = sum(i*i for i in range(n))
        total_time = time.time() - start
        return total, total_time

    if __name__ == "__main__":
        inputs = [ 4 ,10_000_000,20_000_000,30_000_000,40_000_000,50_000_000,60_000_000, 70_000_000]
        process_pool(sum_of_squares, inputs)

    ```
    ____________________________________________________________________________________
    ```
    (base) ray@ip-10-0-54-121:~/default$ python utils/utils.py 
    14 -> time taken 0.0000s
    333333283333335000000 -> time taken 1.6389s
    2666666466666670000000 -> time taken 4.0853s
    21333332533333340000000 -> time taken 5.8987s
    41666665416666675000000 -> time taken 6.1299s
    8999999550000005000000 -> time taken 6.2909s
    114333330883333345000000 -> time taken 7.7595s
    71999998200000010000000 -> time taken 8.6021s
    total time taken 8.6138s

    ```
    ____________________________________________________________________________________
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""#### NOTE: Overall we notice that going from sequential tasks processing to parallel gives us here about 3X speedups.""")
    return


@app.cell
def _():
    import ray
    import logging

    if ray.is_initialized:
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    return (ray,)


@app.cell
def _(inputs, ray, time):
    @ray.remote
    def ray_sum_of_squares(n):
        start = time.time()
        total = sum(i*i for i in range(n))
        total_time = time.time() - start
        return total, total_time

    def ray_execution(fun,inputs):
        start_time = time.time()
        futures = [ray_sum_of_squares.remote(input) for input in inputs]
        ray_result = ray.get(futures)
        print(f"total time take : {time.time() - start_time}s")
        return ray_result

    ray_warmup = ray_execution(ray_sum_of_squares,[100])
    ray_result = ray_execution(ray_sum_of_squares,inputs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## Ray and PyTorch

    In this notebook, we will use `Ray` and `PyTorch` to build and train models for image classification on the CIFAR-10 dataset.

    1. Load the `ResNet` architecture, freeze all layers except the final classification layer, and train only the last layer and evaluate the final model.

    2. Tune the model by experimenting with different hyperparameters.

    3. Serve the Model using Ray Serve and FastAPI.
    """
    )
    return


@app.cell
def _():
    import torch.nn as nn
    import torch
    from torch.utils.data import Subset, DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor
    from torch.optim import SGD
    from torchmetrics import Accuracy

    return Accuracy, CIFAR10, SGD, Subset, ToTensor, nn, torch


@app.cell
def _(CIFAR10, Subset, ToTensor):
    train_data = CIFAR10(root="marimo_notebooks/data",download=True,train=True,transform=ToTensor())
    valid_data= CIFAR10(root="marimo_notebooks/data",download=True,train=False,transform=ToTensor())
    train_sub = Subset(train_data,indices=range(500))
    valid_sub= Subset(valid_data,indices=range(500))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### Train Pytorch Model using Ray Train (WIP)""")
    return


@app.cell
def _():
    from ray.train.torch import TorchTrainer
    return


@app.cell
def _(Accuracy, SGD, SimpleCNN, nn, ray, torch):
    from pathlib import Path
    from ray.train import Checkpoint 


    def ray_simple_torch_train(config):

        # use detected device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # metrics
        accuracy = Accuracy(task="multiclass", num_classes=config["num_classes"]).to(device)
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0.0

        checkpoint_path = Path("/home/ray/default/marimo_notebooks/data/checkpoint")
        checkpoint_path.mkdir(parents=True,exist_ok=True)
        # setup the model and move it to device
        model = SimpleCNN()
        model.to(device)
        model.train()

        # optimiser and loss
        optimiser = SGD(model.parameters(),lr=config["lr"])
        loss_fn = nn.CrossEntropyLoss()

        train_data_shard = ray.train.get_dataset_shard("train")
        train_dataloader = train_data_shard.iter_torch_batches(batch_size=50,dtypes=torch.float32)

        for epoch in range(config["epochs"]):
            for idx, batch in enumerate(train_dataloader):
                x_device , y_device = batch["images"].to(device), batch["labels"].to(torch.long).to(device)
                y_logits = model(x_device)
                y_labels = y_logits.argmax(dim=1)
                loss = loss_fn(y_logits, y_device)
                acc = accuracy(y_labels, y_device)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                train_loss += loss.item()
                train_acc += acc.item()
                num_batches += 1
            train_loss /= num_batches
            train_acc /= num_batches
            torch.save({"model_state_dict": model.state_dict()},str(checkpoint_path/"simple_model.pt"))
            checkpoint =  Checkpoint.from_directory(checkpoint_path)
            #checkpoint arg accepts Checkpoint type
            ray.train.report(checkpoint=checkpoint,metrics={"train_loss":train_loss,"train_acc":train_acc})

        return train_loss, train_acc

    return (ray_simple_torch_train,)


@app.cell
def _(CIFAR10, Subset, ToTensor, ray):
    cifar_train = CIFAR10(root="marimo_notebooks/data/",download=True,train=True,transform=ToTensor())
    cifar_valid = CIFAR10(root="marimo_notebooks/data/",download=True,train=False,transform=ToTensor())
    train_ray_sub = Subset(cifar_train,indices=range(500))
    valid_ray_sub= Subset(cifar_valid,indices=range(500))
    train_ray_ds = ray.data.from_torch(train_ray_sub)
    valid_ray_ds = ray.data.from_torch(valid_ray_sub)

    def prepare_data(row):
        row_data = row["item"]
        images = row_data[0]
        labels = row_data[1]
        return {"images":images,"labels":labels}

    train_ray_data = train_ray_ds.map(prepare_data)
    valid_ray_data = valid_ray_ds.map(prepare_data)

    return train_ray_data, valid_ray_ds


@app.cell
def _(ray_simple_torch_train, train_ray_data):
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig,RunConfig, CheckpointConfig


    datasets ={"train":train_ray_data}

    scaling_config_simple = ScalingConfig(num_workers=4,use_gpu=True)
    run_config_simple = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))
    trainer = TorchTrainer(train_loop_per_worker=ray_simple_torch_train,
                          train_loop_config={"num_classes":10,"lr":0.01,"epochs":10},
                           scaling_config=scaling_config_simple,
                           run_config=run_config_simple,
                           datasets=datasets
                          )

    result_simple = trainer.fit()

    return


@app.cell
def _():
    mo.md(r"""### Load the ResNet architecture and perform batch inference""")
    return


@app.cell
def _(CIFAR10, Subset, ToTensor, ray, valid_ray_ds):
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.transforms import Compose

    weights = ResNet18_Weights.IMAGENET1K_V1
    imagenet_transforms = Compose(transforms=[weights.transforms()])
    inference_model = resnet18(weights=weights)

    cifar_inference_data = CIFAR10(root="marimo_notebooks/data",download=True,transform=ToTensor())
    subset_cifar_data = Subset(cifar_inference_data,indices=range(20))

    valid_dataset_inference= ray.data.from_torch(subset_cifar_data)

    def prepare_for_batch_inference(row):
        row_data = row["item"]
        images = row_data[0]
        labels = row_data[1]
        return {"images":images,"labels":labels}

    valid_ray_inference = valid_ray_ds.map(prepare_for_batch_inference)
    return imagenet_transforms, inference_model, valid_ray_inference, weights


@app.cell
def _(valid_ray_inference):
    single_inference_batch = valid_ray_inference.take_batch(3)
    single_inference_batch

    return


@app.cell
def _(torch):
    class BatchInference:
        def __init__(self,model,weights,transforms):
            self.model = model
            self.weights = weights
            self.transforms = transforms
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.train()

        def __call__(self,batch):
            images = batch["images"]
            labels = batch["labels"]
            # convert them to tensor and then transform them
            transformed_data = [self.transforms(torch.tensor(image).to(self.device)) for image in images]
            stacked_data = torch.stack(transformed_data)
            with torch.inference_mode():
                probabilities = self.model(stacked_data)
                predicted_labels = probabilities.argmax(dim=1)
            return {"predicted_labels": predicted_labels,"images":images}

    return (BatchInference,)


@app.cell
def _(
    BatchInference,
    imagenet_transforms,
    inference_model,
    valid_ray_inference,
    weights,
):
    batch_predictions = valid_ray_inference.map_batches(BatchInference,batch_size=30,fn_constructor_kwargs={"model":inference_model,"weights":weights,"transforms":imagenet_transforms},concurrency=4,num_gpus=1)

    batch_predictions.take(10)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
