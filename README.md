# objectdetection_app
## Object Detection App with Pre-trained R-CNN
## Overview:
This repository contains the main files of the Docker image for an object detection application built on a pre-trained Region-Based Convolutional Neural Network (R-CNN). For a detailed explanation, please check the presentation.

## Features:
1. Utilizes a pre-trained R-CNN for object detection tasks
2. Dockerized deployment for easy portability and scalability
3. Integration with AWS for cloud deployment using Terraform

## Usage:
### To use the object detection app:

1. Ensure Docker is installed on your system.
2. Run the Docker container.
3. Once the Docker container is running, the user is prompted to provide a URL pointing to a JPEG or JPG format image.
4. The application makes predictions on the image, placing colored bounding boxes around detected objects.
5. Objects detected with a confidence score higher than the specified cutoff (default is 0.6) are highlighted. Different colors are assigned for specific object detection tasks, such as green for cars, red for people, blue for trucks, and orange for other known objects.
6. Image with predictions is saved as predicted_image.jpg.

### To download the predicted image:

1. Run the Container: Start the Docker container containing the object detection app by using the command: docker start <container_name>
2. Access Container Terminal: Use the following command to access the terminal inside the Docker container: docker exec -it <container_name> /bin/bash
3. Copy Predicted Image: Once inside the container's terminal, navigate to the directory where the predicted image (predicted_image.jpg) is located. Use the docker cp command to copy the image from the container to your local machine: docker cp <container_name>:/path/to/predicted_image.jpg /path/on/your/local/machine

## Example Prediction 

![Image Input](https://raw.githubusercontent.com/sarok01/objectdetection_app/main/downloaded_image.jpg)

![Predictions](https://raw.githubusercontent.com/sarok01/objectdetection_app/main/predicted_image.jpg)

