import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# -------------------------------
# GradCAM Implementation
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with the model and the target layer.
        Args:
            model: The trained neural network.
            target_layer: The convolutional layer from which we capture activations and gradients.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Forward hook: capture activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
        # Backward hook: capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        handle_fwd = self.target_layer.register_forward_hook(forward_hook)
        handle_bwd = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fwd, handle_bwd])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate a GradCAM heatmap for the given input tensor.
        Args:
            input_tensor: Preprocessed input image tensor with batch dimension.
            class_idx: (Optional) The target class index. If None, uses the predicted class.
        Returns:
            cam: A numpy array containing the normalized heatmap.
            output: The raw model output.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # Forward pass

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward()  # Backward pass

        # Retrieve gradients and activations from the target layer
        gradients = self.gradients[0]      # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]

        # Global average pooling on gradients: compute channel-wise weights
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: [C]
        # Create the cam tensor on the same device as activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), output

# -------------------------------
# Model Loading and Setup
# -------------------------------
@st.cache_resource
def load_model():
    # Device configuration: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define class names (order must match training)
    class_names = [
        "aeroplane",
        "flying_rocket",
        "helicopter",
        "paragliding",
        "parasailing",
        "sky diving"
    ]
    num_classes = len(class_names)
    # Use ResNet50 to match the checkpoint architecture
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model_path = "../models/aerial_activity_detector_robust_hd2.pth"  # Adjust path if necessary
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
    model = model.to(device)
    model.eval()
    return model, device, class_names

model, device, class_names = load_model()

# -------------------------------
# Preprocessing Transformation
# -------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Aerial Activity Detector with GradCAM")
st.write("Upload an image to see the prediction along with GradCAM visualization.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        orig_image = Image.open(uploaded_file).convert("RGB")
        st.image(orig_image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")
    else:
        # Preprocess the image for the model
        input_tensor = preprocess(orig_image).unsqueeze(0).to(device)

        # -------------------------------
        # Select the Target Layer for GradCAM
        # -------------------------------
        # For ResNet50, we choose the last conv layer of the last block.
        target_layer = model.layer4[2].conv3  # Adjust if needed
        gradcam = GradCAM(model, target_layer)

        # Generate GradCAM heatmap and prediction
        cam, output = gradcam.generate(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        predicted_class = class_names[pred_idx.item()]
        confidence_value = confidence.item() * 100

        # Prepare the heatmap overlay
        cam_resized = cv2.resize(cam, (orig_image.width, orig_image.height))
        heatmap = np.uint8(255 * cam_resized)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original PIL image to OpenCV format (BGR)
        orig_np = np.array(orig_image)
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        
        # Blend the heatmap with the original image to create an overlay
        overlay = cv2.addWeighted(orig_np, 0.5, heatmap_color, 0.5, 0)
        
        # Threshold the heatmap to create a binary mask for salient regions
        ret, mask = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{predicted_class}: {confidence_value:.2f}%"
            cv2.putText(overlay, text, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            text = f"{predicted_class}: {confidence_value:.2f}%"
            cv2.putText(overlay, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert overlay from BGR to RGB for Streamlit display
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        st.image(overlay_rgb, caption="Prediction with GradCAM", use_column_width=True)
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence_value:.2f}%")
        
        # Clean up GradCAM hooks
        gradcam.remove_hooks()
