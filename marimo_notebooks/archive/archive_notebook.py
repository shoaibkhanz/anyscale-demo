import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import ray 
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Compose
    from torchvision.models import resnet152, ResNet152_Weights
    from torchmetrics.classification import Accuracy
    from torchvision.datasets import CIFAR10
    import matplotlib.pyplot as plt
    import marimo as mo
    from torch.utils.data import Subset


@app.cell
def _():
    cifar_train1 = CIFAR10(root="marimo_notebooks/data/", download=True,
                           transform=ToTensor(),
                           train=True)
    cifar_valid1= CIFAR10(root="marimo_notebooks/data/", download=True,
                          transform=ToTensor(),
                          train=False
                         )

    return cifar_train1, cifar_valid1


@app.cell
def _(cifar_train1, cifar_valid1):
    train_sub = Subset(dataset=cifar_train1,indices=range(100))
    valid_sub = Subset(dataset=cifar_valid1,indices=range(100))

    return train_sub, valid_sub


@app.function
def convert_cifar_to_ray_data(dataset):
    data_list = []
    for data, label in dataset:
        data_list.append({"image":data, "label": label})
    ray_dataset = ray.data.from_items(data_list)
    return ray_dataset


@app.cell
def _(train_sub, valid_sub):
    train_ds1 = convert_cifar_to_ray_data(train_sub)
    valid_ds1 = convert_cifar_to_ray_data(valid_sub)

    return (train_ds1,)


@app.cell
def _(train_ds1):
    sample_batch= train_ds1.take_batch(5)
    sample_batch
    return


@app.cell
def _():
    imagenet_transform = ResNet152_Weights.IMAGENET1K_V1.transforms
    transforms = Compose([imagenet_transform()])
    return (transforms,)


@app.cell
def _(transforms):
    class ResNetInference:
        def __init__(self):
            self.weights = ResNet152_Weights.IMAGENET1K_V1
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = resnet152(weights=self.weights)
            self.model.to(self.device)
            self.model.eval()

        def __call__(self,batch):
            image_batch = batch["image"]
            transform_batch = [transforms(torch.tensor(row)) for row in image_batch]
            inputs = torch.stack(transform_batch).to(self.device)
            with torch.inference_mode():
                pred = self.model(inputs)
                pred_label = pred.argmax(dim=1).cpu().numpy()
            return {"pred": pred_label,"label":batch["label"]}

    return (ResNetInference,)


@app.cell
def _(ResNetInference, train_ds1):
    pred = train_ds1.map_batches(ResNetInference,batch_size=50,concurrency=1,num_gpus=1)
    return (pred,)


@app.cell
def _(pred):
    pred.take_all()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
