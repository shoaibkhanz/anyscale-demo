import marimo

__generated_with = "0.14.15"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell
def _():
    mo.md(r"## Agenda")
    return


@app.cell
def _():
    mo.md(
        f"""
    ## Building Scalable and Performant Solutions

    {mo.image("/home/ray/default/marimo_notebooks/resources/ray.png")}

    ## What is RAY ?
    Ray is an open-source framework to build and scale ML and Python applications. It consists of easy to use APIs that allows the developer to run their workloads in heterogenous compute (CPUs and GPUs), supports any data type and model architecture, scaling to thousands of GPUs while keeping a stable utilisation across all your compute.

    > Anyscale is a commercial provider that offers a managed platform built on top of Ray with further optimisations and new robust features such as [Ray Turbo](https://www.anyscale.com/product/platform/rayturbo) and others. It abstracts away the operational complexity of deploying and managing Ray clusters, enabling teams to focus on developing scalable applications without worrying about infrastructure.


    {mo.image("/home/ray/default/marimo_notebooks/resources/anyscale_features.png")}
    """
    )
    return


@app.cell
def _():
    mo.md(
        rf"""
    ## Ray Ecosystem 
    {mo.image("/home/ray/default/marimo_notebooks/resources/ray-libraries.png",height=400, width=550)}

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


@app.cell
def _():
    mo.md(
        f"""
    # Scaling Python Applications
    ## Motivating Example

    We will start with a motivating example in pure python and then introduce Ray to better present the challenge Ray is trying to solve for modern python applications. To understand this well, we must be aware of some fundamental concepts, we wont dive deep into them but briefly state them such that we can grasp the high-level concept.

    ### Concurency vs Parallelism ?
    Lets first understand the challenges we face executing compute tasks in Python and its ability to scale. There is a distinction between running tasks in a concurent vs in a parallel fashion, this distintion also causes some confusion if not understood well. 

    - **Concurrent Tasks**: When tasks are progressing at the sametime but not necesarily executing simulataneously, such tasks are said to be concurrent tasks. *for e.g. Imagine sitting at a restaurant where the chef takes multiple orders but switches between them as he prepares them.*
    - **Parallel Tasks**: When tasks are executed at the same time, such tasks are called to be parallel tasks. *for e.g Imagine the same restaurant but now with multiple chefs each preparing the order.*

    {mo.hstack([
        mo.image("/home/ray/default/marimo_notebooks/resources/concurrent-burgers-03.png", height=400, width=800),
        mo.image("/home/ray/default/marimo_notebooks/resources/parallel-burgers-01.png", height=400, width=800),
    ])}
    """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
        ### Threads and Processes
    
    
        Let’s understand how Python runs code when we compute something like `ssq(n)` (sum of squares), and what it means to use **threads** vs **processes** — and eventually, how **Ray** takes this to a distributed scale.
    
    
        #### The Basics: Threads and Processes in Python
    
        - When you run a Python function like `ssq(n)`, it executes in the **main thread** of a **process**.
        - A **process** has:
          - Its own **Python interpreter**
          - Its own **memory space**
          - One or more **threads** that run code
    
        - A **thread** shares memory and interpreter state with other threads in the same process.
    
    
        #### Multiple Threads
    
        - If you run multiple threads in Python (e.g. calling `ssq(10)`, `ssq(20)` in parallel threads):
          - They **share the same memory**
          - But due to the **GIL (Global Interpreter Lock)**, only **one thread runs Python bytecode at a time**
    
    
        #### The GIL Problem
    
        - Python’s GIL prevents true parallel execution of threads for CPU-bound tasks.
        - This means if you want to **truly parallelize** CPU-heavy functions like `ssq(n)`, you need to use **multiple processes**, not threads.
        """
    )
    return


@app.cell
def _():
    mo.md(
        r"""
        #### Processes to the Rescue
    
        - A **process** has its own interpreter and memory.
        - So multiple processes **can run in parallel** on different CPU cores.
        - This allows true concurrency for CPU-bound functions.
    
        But:
        - You now need a way to **send work to processes**
        - And a way to **gather the results intelligently**
    
    
        #### Enter Ray
    
        Now imagine:
        - Running `ssq(n)` across **hundreds or thousands of machines**
    
        - Automatically managing:
            - Parallel execution
            - Result collection
            - Scheduling and retries
    
        This is where **Ray** comes in. Ray takes care of **distributed execution**, **parallelism**, and **result aggregation** at scale.
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
    def thread_pool(func,inputs):
        start_time = time.time()
        with ThreadPoolExecutor() as thread_executor:
            futures = {thread_executor.submit(func,input) for input in inputs}
            print(futures)

            for future in as_completed(futures):
                result, execution_time = future.result()
                print(f"{result} -> time taken {execution_time:.4f}s")
        print(f"total time taken {(time.time() - start_time):.4f}s")

    thread_pool(sum_of_squares,inputs)

    return


@app.cell
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


@app.cell
def _():
    mo.md(r"#### NOTE: Overall we notice that going from sequential tasks processing to parallel gives us here about 3X speedups.")
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


@app.cell
def _():
    mo.md(
        r"""
        ## Ray and PyTorch
    
        In this notebook, we will use `Ray` and `PyTorch` to build and train models for image classification on the CIFAR-10 and MNIST datasets.
    
        1. Train and evaluate a simple CNN model using native PyTorch on CPU/GPU.
    
        2. Extend the same architecture to use `Ray Train` for distributed training.
    
        3. Load the `ResNet` architecture and perform batch inference.
    
        4. Apply the following model training configurations for ResNet:
    
               4.1. Freeze all layers except the final classification layer, and train only the last layer.
    
               4.2. Freeze selected layers (e.g., last few layers), and fine-tune the model.
    
               4.3. Train the entire model from scratch without using any pretrained weights.
    
               4.4. Modify the input layer of `ResNet` to accept grayscale (1-channel) images, and train the model on MNIST.
    
        5. Train and evaluate the ResNet model using different training and data-loading configurations:
    
               5.1. Using PyTorch `DataLoader` with `Ray Train`
    
               5.2. Using `Ray Data` with `Ray Train`
        """
    )
    return


@app.cell
def _():
    mo.md(r"### Train and evaluate a simple CNN model using native PyTorch on CPU/GPU")
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

    return Accuracy, CIFAR10, DataLoader, SGD, Subset, ToTensor, nn, torch


@app.cell
def _(nn):
    class SimpleCNN(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64, output_dim=10):
            super().__init__()

            self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  
            )

            self.conv_block2 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim * 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2), 
                nn.Dropout(0.25)
            )

            self.conv_block3 = nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim * 4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Flatten()
            )

            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(hidden_dim * 4, output_dim)
            )

        def forward(self, x):
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return self.classifier(x)


    return (SimpleCNN,)


@app.cell
def _(Accuracy, CIFAR10, DataLoader, Subset, ToTensor):
    train_data = CIFAR10(root="marimo_notebooks/data",download=True,train=True,transform=ToTensor())
    valid_data= CIFAR10(root="marimo_notebooks/data",download=True,train=False,transform=ToTensor())
    train_sub = Subset(train_data,indices=range(500))
    valid_sub= Subset(valid_data,indices=range(500))
    train_dataloader = DataLoader(train_sub, batch_size=50,shuffle=True)
    valid_dataloader = DataLoader(valid_sub, batch_size=50,shuffle=False)


    def pytorch_simple_train(epochs, model,dataloader,device,optimiser,loss_fn):
        model.to(device)
        model.train()
        train_loss =0.0 
        train_acc =0.0 
        batch_train_loss = []
        batch_train_acc= []
        accuracy = Accuracy(task="multiclass",num_classes=10).to(device)
        for epoch in range(epochs):
            for idx, (x,y) in enumerate(dataloader):
                x_device, y_device = x.to(device),y.to(device)
                y_logits = model(x_device)
                y_labels = y_logits.argmax(dim=1)
                loss = loss_fn(y_logits,y_device)
                acc = accuracy(y_labels,y_device)
                train_loss +=loss.item()
                train_acc +=acc.item()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            train_loss /= len(dataloader)
            train_acc/= len(dataloader)
            batch_train_loss.append(train_loss)
            batch_train_acc.append(train_acc)
        return  batch_train_loss, batch_train_acc

    return pytorch_simple_train, train_dataloader


@app.cell
def _(SGD, SimpleCNN, nn, pytorch_simple_train, train_dataloader):
    simple_model = SimpleCNN(input_dim=3,hidden_dim=64,output_dim=10)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = SGD(simple_model.parameters(),lr=0.1)


    train_loss, train_acc= pytorch_simple_train(epochs=10,model=simple_model,dataloader=train_dataloader,device="cpu",loss_fn=loss_fn,optimiser=optimiser)

    return train_acc, train_loss


@app.cell
def _(train_acc, train_loss):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,2,figsize=(15,5))
    ax = axes.flatten()
    ax[0].plot(train_loss,"b--")
    ax[0].set_title("Training Loss")
    ax[1].plot(train_acc,"b--")
    ax[1].set_title("Training Accuracy")
    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    mo.md(r"### Simple Pytorch Model using Ray Train (WIP)")
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
    mo.md(r"### Load the ResNet architecture and perform batch inference")
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
