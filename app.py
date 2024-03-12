# Load image

from PIL import Image
import requests
from io import BytesIO
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch import load
from pycocotools.coco import COCO
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

def download_image(image_url):
    try:
        # Send a GET request to the URL
        response = requests.get(image_url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the image using PIL
            image = Image.open(BytesIO(response.content))
            
            # Save the image
            image.save("./downloaded_image.jpg")
        else:
            print(f"Failed to retrieve the image. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Load Images as Pillow Images
            
def load_image(image_path):    
    try:
        target_image = Image.open(image_path)
        return target_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

# Transform
    
def convert_to_PIL(target_image):
    try:         
        target_image_tensor_int = pil_to_tensor(target_image)
        #print(target_image_tensor_int.shape)
        return target_image_tensor_int
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def add_batchdim(target_image_tensor_int):
    try:
        target_image_tensor_int = target_image_tensor_int.unsqueeze(dim=0)
        #print(target_image_tensor_int.shape)
        return target_image_tensor_int

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_imagerep_from_int_to_float(target_image_tensor_int):
    try: 
        #print(target_image_tensor_int.min(), target_image_tensor_int.max())

        target_image_tensor_float = target_image_tensor_int / 255.0

        #print(target_image_tensor_float.min(), target_image_tensor_float.max())
        #print(target_image_tensor_float)

        return target_image_tensor_float

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Load Pretrained Model - (Faster R-CNN with ResNet50 Backbone) 
    
def load_pretrained_model(weights_path):
    try:
        object_detection_model = fasterrcnn_resnet50_fpn(weights= load(weights_path))

        return object_detection_model

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# Make predictions
    
def make_pred(object_detection_model, target_image_tensor_float, confidence):
    try:
        target_image_preds = object_detection_model(target_image_tensor_float)

        target_image_preds[0]["boxes"] = target_image_preds[0]["boxes"][target_image_preds[0]["scores"] > confidence]
        target_image_preds[0]["labels"] = target_image_preds[0]["labels"][target_image_preds[0]["scores"] > confidence]
        target_image_preds[0]["scores"] = target_image_preds[0]["scores"][target_image_preds[0]["scores"] > confidence]

        #print(target_image_preds)
        return target_image_preds

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

# Class mapping
    
def class_mapping(annFile, target_image_preds):
    try:
        coco=COCO(annFile)
        target_image_labels = coco.loadCats(target_image_preds[0]["labels"].numpy())

        #print(target_image_labels)
        return target_image_labels

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  
    
def visualize_bounding_boxes(target_image_labels, target_image_preds, target_image_tensor_int):
    try:
        target_image_annot_labels = ["{}-{:.2f}".format(label["name"], prob) for label, prob in zip(target_image_labels, target_image_preds[0]["scores"].detach().numpy())]

        target_image_output = draw_bounding_boxes(image=target_image_tensor_int[0],
                                boxes=target_image_preds[0]["boxes"],
                                labels=target_image_annot_labels,
                                #colors=["red" if label["name"]=="person" else "green" for label in target_image_labels],
                                colors = ["red" if label["name"] == "person" else "blue" if label["name"] == "truck" else "green" if label["name"] == "car" else "orange" for label in target_image_labels],
                                width=10,
                                fill=True
                                )

        #print(target_image_output.shape)
        return target_image_output
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def save_image(target_image_output, predicted_image_path):
    try:
        predicted_image = to_pil_image(target_image_output)

        predicted_image.save(predicted_image_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def main():
    # Ask the user to input the image URL
    image_url = input("Enter the URL of the image: ")

    # Call the function to download the image
    download_image(image_url)

    # Load the downloaded image
    target_image = load_image("downloaded_image.jpg")

    # Transform

    target_image_tensor_int = convert_to_PIL(target_image)

    target_image_tensor_int = add_batchdim(target_image_tensor_int)

    target_image_tensor_float = convert_imagerep_from_int_to_float(target_image_tensor_int)

    # Load ML model

    weights_path = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

    object_detection_model = load_pretrained_model(weights_path)

    object_detection_model.eval(); ## Setting Model for Evaluation/Prediction

    # Make predictions

    confidence = 0.6
    target_image_preds = make_pred(object_detection_model, target_image_tensor_float, confidence)

    # Class mapping

    annFile='annotations/annotations/instances_val2017.json'

    target_image_labels = class_mapping(annFile, target_image_preds)

    # Visualize Bounding Boxes On Original Images

    target_image_output = visualize_bounding_boxes(target_image_labels, target_image_preds, target_image_tensor_int)

    # Convert back to PIL and save 

    predicted_image_path = "predicted_image.jpg"
    save_image(target_image_output, predicted_image_path) 




if __name__ == "__main__":
    main()

# url= https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg
# url2 = https://images.pexels.com/photos/1600757/pexels-photo-1600757.jpeg?auto=compress&cs=tinysrgb&w=800


