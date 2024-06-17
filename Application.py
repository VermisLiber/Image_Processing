import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Loading the pre-trained GoogLeNet model with explicit settings
model = models.googlenet(weights=None, aux_logits=True, init_weights=True, num_classes=3)  # Adjust num_classes as per your pretrained model

# Loading the state dictionary saved from the original model training
state_dict = torch.load(r"C:\Users\PRAJNA RANGANATH\source\repos\CNN\googlenet_model.pth", map_location=torch.device('cpu'))

# Filtering out keys related to aux branches if they are causing mismatches
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux')}

# Modifying the fc weights and biases to match the current model's fc layers
fc_weight = state_dict['fc.weight']
fc_bias = state_dict['fc.bias']

# Adjusting the shape of fc weight and bias according to the current model's fc layer
# Assuming current model's fc layer has 3 output classes
expected_fc_weight_shape = model.fc.weight.shape
expected_fc_bias_shape = model.fc.bias.shape

if fc_weight.shape != expected_fc_weight_shape:
    # Resizeing fc_weight to match expected_fc_weight_shape
    fc_weight = fc_weight[:expected_fc_weight_shape[0], :]

if fc_bias.shape != expected_fc_bias_shape:
    # Resizeing fc_bias to match expected_fc_bias_shape
    fc_bias = fc_bias[:expected_fc_bias_shape[0]]

# Updateing the state_dict with modified fc weight and bias
state_dict['fc.weight'] = fc_weight
state_dict['fc.bias'] = fc_bias

# Loading the modified state dictionary into the model
model.load_state_dict(state_dict, strict=False)

# Setting the model to evaluation mode
model.eval()

# Defining transformations - resize and normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch

# Defining class labels 
class_names = {
    0: "Car",
    1: "Cat",
    2: "Man"
}

# Function to predict the class name of an image
def predict_image(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
    predicted_index = np.argmax(output.numpy())
    predicted_label = class_names.get(predicted_index, "Unknown")
    return predicted_label

# Path to the image
image_path = r"C:\Users\PRAJNA RANGANATH\OneDrive\Desktop\Image Processing\Dataset\Checking\Man\010.jpg"

# Predicting the class of the image
predicted_label = predict_image(image_path)
print(f"Predicted class: {predicted_label}")
