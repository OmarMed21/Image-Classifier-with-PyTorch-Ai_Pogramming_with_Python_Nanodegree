import numpy as np
import torch ## pytorch the main package 
from PIL import Image ## Pillow Package
import json ## when we're willing to import json files and use it during our code
import argparse
from torchvision import datasets, transforms, models
from train import load_checkpoint, process_image

parser = argparse.ArgumentParser()

parser.add_argument('image', type=str, default='test/16/image_06670.jpg', help='input image path')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='trained model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='top k most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to actual names')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for inference')

in_arg = parser.parse_args()

def predict(image_path, checkpoint, device, topk=5):
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    
    model, optimizer = load_checkpoint(checkpoint)
    model.to(to_device)
    model.eval()
    image_tensor = process_image(image_path)

    with torch.no_grad():
        image_tensor = image_tensor.to(to_device)
        image_tensor.unsqueeze_(0)
        image_tensor.float()
        logps = model.forward(image_tensor)
        ps = torch.exp(logps)
        top_ps, top_class_idx = ps.topk(topk, dim=1)
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_classes = [class_to_idx_inverted[i] for i in top_class_idx.cpu().numpy()[0]]
    top_probs = top_ps.cpu().numpy()[0]
    
    return top_probs, top_classes

def main():
    probs, classes = predict(in_arg.image, in_arg.checkpoint, in_arg.gpu, topk=in_arg.top_k)
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    flower_classes = [cat_to_name[j] for j in classes]
    print(flower_classes)
    print(probs)
    

if __name__ == '__main__':
    main()
