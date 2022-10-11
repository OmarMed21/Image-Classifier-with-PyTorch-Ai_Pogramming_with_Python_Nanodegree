import torch ## ## pytorch the main package 
from torch import optim, nn ## Neural Network
import torch.nn.functional as F ## Applies a 3D convolution over an input image composed of several input planes.
from torchvision import datasets, transforms, models
from PIL import Image ## Pillow as like as OpenCV , used for Image Processing
import argparse ## makes it easy to write user-friendly command-line interfaces.

## create Neural Network with the traditional ways
class Classifier(nn.Module):
    ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
    def __init__(self, input_size, output_size, hidden_layers, drop_p= .2):
        super().__init__
        ## input hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """
        Fast Forward Neural Network .. passing each input after hidden layers to 
        activation function of Relu then it ends with sigmoid or SoftMax

        """
        ## loop over all hidden layers
        for each in self.hidden_layers:
            ## create relu using Functional Neural Network
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        ## the last layer is SoftMax()
        return F.log_softmax(x, dim=1)
        
def validation_loss(model, test_loader, criterion):
    acc = 0.0 ## stands for intial accuracy
    test_loss = 0.0

    for imgs, lbs in test_loader:
        ## need to flatten the images into 784 long vector
        imgs = imgs.resize_(imgs.size()[0], 784)
        out = model.forward(imgs) ## perform fast forward on the neural network of images 
        
        # -----------> DON'T UNDERSTAND THIS <-----------
        test_loss += criterion(out, lbs).item()
        # -----------> DON'T UNDERSTAND THIS <-----------

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(out)

        # -----------> DON'T UNDERSTAND THIS <-----------
        # Class with highest probability is our predicted class, compare with true label
        equality = (lbs.data == ps.max(1)[1])
        # -----------> DON'T UNDERSTAND THIS <-----------

        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        acc += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, acc

def train(model, train_loader, test_loader, criterion, optimizer, epochs=5, print_every=40):
    steps = 0
    running_loss = 0
    ## just get started with n epochs ,, we can  modify it easily
    for e in range(epochs):
        ## model Training
        model.train()
        ## just like validation_loss() function loop over the train_loader this time not the test
        for imgs, lbs in train_loader:
            steps += 1

            ## again need to flatten the images into 784 long vector    
            imgs.resize_(imgs.size()[0], 784)

            ## In PyTorch, for every mini-batch during the training phase,
            ## we typically want to explicitly set the gradients to zero before starting to do backpropragation
            ## (i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes.
            ## This accumulating behaviour is convenient while training RNNs or when we want to compute the gradient of the loss summed over multiple mini-batches.
            ## So, the default action has been set to accumulate (i.e. sum) the gradients on every loss.backward() call.
            optimizer.zero_grad()
            
            ## go through the network normally Fast forward
            output = model.forward(imgs)

            ## claculate the loss by backpropgation
            ## as we've prepared the optimizer above
            loss = criterion(output, lbs)
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item()

            ## print every 40 mini batches
            if steps % print_every == 0:
                ## model.eval() is a kind of switch for some specific layers/parts
                ## of the model that behave differently during training and inference (evaluating) time.
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation_loss(model, test_loader, criterion)

                    test_loss, accuracy = validation_loss(model, test_loader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

def create_model(hidden_units=1024, learnrate = 0.001, device='gpu'):
    ## the same step for every time we're trying to train an image classifier :) ==> MAKE SURE THAT IT RUNS ON GPU
    to_device = torch.device('cuda' if torch.cuda.is_available() and device=='gpu' else 'cpu')
    ## I myself have used densenet data so i've only choosed it
    model = models.densenet121(pretrained=True)
    in_size = 1024 ## it's long
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = Classifier(in_size, 102, [hidden_units])

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    model.to(to_device)
    return model, optimizer

def load_data():
    """
    Load the dataset you've already used... in our case the Flowers with the three folders of train,valid,test
    """
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir =  'test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data
        
def save_checkpoint(model, optimizer, path='checkpoint.pth', hidden_units=1024, epochs=5):
    checkpoint = {'arch': 'densenet121',
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state': optimizer.state_dict(),
                  'epochs': epochs}

    torch.save(checkpoint, path)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model, optimizer = create_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return model, optimizer

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_img = Image.open(image)
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    img_tensor = preprocess(pil_img)
    return img_tensor

## Type some lines in the command interface
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, default='test/9/image_06410.jpg', help='input image path')
parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='trained model checkpoint')
parser.add_argument('--top_k', type=int, default=5, help='top k most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to actual names')
parser.add_argument('--gpu', type=str, default='gpu', help='use GPU for inference')
in_arg = parser.parse_args() ## parse_args() will return an object with two attributes, integers and accumulate

## Now it's time to check out the results
def main():
    trainloader, validloader, testloader, train_data = load_data()
    model, optimizer = create_model(arch=in_arg.arch, hidden_units=in_arg.hidden_units, learnrate=in_arg.learning_rate, device=in_arg.gpu)
    criterion = nn.NLLLoss()
    train(trainloader, validloader, model, optimizer, criterion, in_arg.gpu, in_arg.epochs)
    model.class_to_idx = train_data.class_to_idx
    save_checkpoint(model, optimizer, path=in_arg.save_dir, arch=in_arg.arch, hidden_units=in_arg.hidden_units, epochs=in_arg.epochs)
    print("====Training completed.====")
    

if __name__ == '__main__':
    main()






