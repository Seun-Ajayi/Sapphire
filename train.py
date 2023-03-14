from futility import data_loader
import torch
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict


def get_parser() -> argparse.ArgumentParser:
    """
    parse command line arguments

    returns:
        parser - ArgumentParser object
    """

    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument(
        '--data_dir',
        type=str,
        default= 'flower_data',
        help='Root directory of the data files'
)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for the model, default=0.001'
)
    parser.add_argument(
        '--structure',
        type=str,
        default="vgg16",
        help='Model architecture from torchvision.models'
)
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of epochs for model training, default: 3'
)
    parser.add_argument(
        '--output_file_pth',
        type=str,
        default="checkpoint.pth",
        help='path to the file to save the model parameters'
)
    parser.add_argument(
        '--device',
        type=str.lower,
        default="gpu",
        help='Device type to run the modelling, default: GPU'
)

    return parser


def data_loader( data_dir: str="flower_data"):


    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    train_transforms = transforms.Compose (
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    test_transforms = transforms.Compose (
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    valid_transforms = transforms.Compose (
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]
    )

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True) 
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)


    return trainloader, testloader, validloader, train_data 

def nn_setup(structure="vgg16", lr=0.001):

    if structure == "vgg16":
        model = models.vgg16(pretrained = True)
        input_shape = 25088
    else:
        model = models.alexnet(pretrained=True)
        input_shape = 9216

    for params in model.parameters():
        params.requires_grad=False

    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_shape, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.3)),
        ('fc2', nn.Linear(4096, 2048)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.3)),
        ('fc3', nn.Linear(2048, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model = model.to("cuda")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion, optimizer, input_shape



def train_model(
        trainloader, 
        validloader, 
        model, 
        criterion, 
        optimizer, 
        train_data,
        output_file_pth,
        epochs,
        structure,
        input_shape,
        lr,
        device="gpu"
    ):
    
    if device == "gpu":
        device = "cuda"
    else:
        device = "cpu"
    print_every = 5
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to("device"), labels.to("device")
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if steps & print_every == 0:
            model.eval()
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to('device'), labels.to('device')

                    logs_ps = model.forward(inputs)
                    batch_loss = criterion(logs_ps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logs_ps)
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Loss: {running_loss/print_every:.3f}.. "
                    f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                    f"Accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

    
    # TODO: Save the checkpoint 

    model.class_to_idx = train_data.class_to_idx
    torch.save({
        "input_size": input_shape,
        "output_size": 102,
        "structure": structure,
        "learning_rate": lr,
        "epochs": epochs,
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx
    }, output_file_pth)



if __name__ == "__main__":
    parser = get_parser()
    params, _ = parser.parse_known_args()

    trainloader, testloader, validloader, train_data = data_loader(params.data_dir)
    model, criterion, optimizer, input_shape = nn_setup(structure=params.structure, lr=params.lr)

    train_model(
        trainloader=trainloader, 
        validloader=validloader, 
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_data=train_data,
        output_file_pth=params.output_file_pth,
        epochs=params.epochs,
        device=params.device,
        structure=params.structure,
        input_shape=input_shape,
        lr=params.lr
    )

