import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torch import nn
import timm

# Define the model
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b3', pretrained=False, num_classes=7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits

# Load the model
model = FaceModel()
model.load_state_dict(torch.load('best-weights.pt', map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing transformations
transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names for the facial expressions
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize camera
cap = cv2.VideoCapture(0)

def predict_frame(frame):
    # Convert frame from BGR (OpenCV format) to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transformations to the image
    img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
    
    # Run the image through the model
    with torch.no_grad():
        logits = model(img_tensor)
        probs = nn.Softmax(dim=1)(logits)
        return probs.squeeze().tolist()  # Return probabilities as a list

def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale=0.8, font_thickness=2, box_color=(0, 0, 0), text_color=(255, 255, 255)):
    # Get the size of the text box
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Set the position for the text box
    x, y = position
    box_coords = ((x, y - text_height - 10), (x + text_width + 10, y))
    
    # Draw the background rectangle
    cv2.rectangle(frame, box_coords[0], box_coords[1], box_color, -1)  # Fill the rectangle with box_color

    # Draw the text
    cv2.putText(frame, text, position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Get prediction from the model
    probs = predict_frame(frame)
    
    # Find the predicted class
    predicted_class = classes[probs.index(max(probs))]
    confidence = max(probs)
    
    # Display the predicted class on the frame
    label = f"{predicted_class} ({confidence:.2f})"
    draw_text_with_background(frame, label, (10, 50))

    # Display all probabilities
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        text = f"{cls}: {prob:.2f}"
        draw_text_with_background(frame, text, (10, 100 + i * 30))

    # Display probability distribution for debugging
    print(f"Probability Distribution: {dict(zip(classes, probs))}")

    # Show the frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
