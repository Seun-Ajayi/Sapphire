import torch
import json
import argparse
import numpy as np
from train import nn_setup
from torchvision import transforms
from PIL import Image


def get_parser() -> argparse.ArgumentParser:
    """
    parse command line arguments

    returns:
        parser - ArgumentParser object
    """

    parser = argparse.ArgumentParser(description='Predict Script')
    parser.add_argument(
        '--image_path',
        type=str,
        help='path of image to be predicted'
)
    parser.add_argument(
        '--model_path',
        type=str,
        default="checkpoin.pth",
        help='Path to directory model is saved'
)
    parser.add_argument(
        '--device',
        type=str.lower,
        default="gpu",
        help='Device type to run the modelling, default: GPU'
)
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='Number of top classes to be considered, default=5'
)
    parser.add_argument(
        '--json_filepath',
        type=str,
        default="cat_to_name.json",
        help='Path of file containing the names of the flowers'
)

def load_model(path):

  checkpoint = torch.load(path)
  model,*_ = nn_setup()
  model.class_to_idx = checkpoint["class_to_idx"]
  model.load_state_dict = checkpoint["state_dict"]

  return model 

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = img_transforms(img_pil)

    return image

def predict(image_path, model, topk=5, device="gpu"):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if device == "gpu":
       device = "cuda"
    else:
       device = "cpu"


    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
      logs_ps = model.forward(img.to("cuda"))

    prob = torch.exp(logs_ps).data

    return prob.topk(topk)


if __name__ == "__main__":
   parser = get_parser()
   params, _ = parser.parse_known_args()

   with open(params.json_filepath, "r") as f:
      cat_to_name = json.load(f)

   model = load_model(params.model_path)
   ps = predict(params.image_path, model, params.topk, params.device)

   x = np.array(ps[0][0])
   y = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]

   for i in range(params.topk):
     print("{}. Predicting: ___{}___ with probability: {:.2f}%.".format(i+1, y[i], x[i]*100))




