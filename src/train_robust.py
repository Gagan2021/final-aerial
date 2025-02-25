# train_robust.py

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# --------------------------------
# (1) Mixup Helper Function
# --------------------------------
def mixup_data(x, y, alpha=0.2):
    """
    Perform mixup augmentation on a batch of data.
    
    Args:
        x (Tensor): Input batch of images.
        y (Tensor): Labels (class indices) for the batch.
        alpha (float): Mixup hyperparameter. If 0, no mixup is applied.
    
    Returns:
        mixed_x, y_a, y_b, lam: Mixed inputs, original labels, shuffled labels, and mixup lambda.
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# --------------------------------
# (2) Training Function (Mixup + Label Smoothing)
# --------------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, 
                num_epochs=50, mixup_alpha=0.2):
    """
    Train the model using both training and validation phases.
    Uses mixup augmentation (only in training) and label smoothing via the loss function.
    
    Args:
        model: The neural network model (ResNet, etc.).
        dataloaders: Dict with 'train' and 'val' DataLoaders.
        criterion: Loss function (with label smoothing).
        optimizer: Optimizer (Adam, SGD, etc.).
        scheduler: Learning rate scheduler (CosineAnnealingLR here).
        device: Device (CPU or GPU) to run on.
        num_epochs (int): Number of epochs to train.
        mixup_alpha (float): Mixup parameter. 0 = no mixup.
    
    Returns:
        model: The best model (weights) based on validation loss.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Only apply mixup during training
                if phase == 'train' and mixup_alpha > 0:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    
                    # Predictions
                    _, preds = torch.max(outputs, 1)
                    # Count partial correctness for each label in mixup
                    correct_mix = (preds == targets_a).float() * lam + (preds == targets_b).float() * (1 - lam)
                    correct_count = correct_mix.sum()
                else:
                    # Normal forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    correct_count = torch.sum(preds == labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += correct_count

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Check for best validation
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            # Step the scheduler after the training phase
            if phase == 'train':
                scheduler.step()
        
        print()

    print(f"Best Validation Loss: {best_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# --------------------------------
# (3) Grad-CAM Visualization
# --------------------------------
def visualize_gradcam(model, device, image_path, transform, 
                      target_layer=None, target_category=None, 
                      output_path="gradcam_result.jpg"):
    """
    Apply Grad-CAM to visualize model focus on a single image.

    Args:
        model: Trained PyTorch model (e.g., ResNet18).
        device: 'cpu' or 'cuda'.
        image_path: Path to the image for visualization.
        transform: Transform pipeline for the image (same as validation).
        target_layer: The layer to target for Grad-CAM (e.g., model.layer4[-1] for ResNet).
        target_category: If None, uses the predicted class as the target.
        output_path: File name to save the visualization result.
    """
    import numpy as np
    import cv2
    from PIL import Image
    # Make sure you have pytorch_grad_cam installed:
    # pip install pytorch_grad_cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()

    # 1. Load and transform the image
    img_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 2. Choose a target layer (if not provided)
    if target_layer is None:
        # For ResNet18, model.layer4 is the final block
        target_layer = model.layer4[-1]  # The last block of layer4

    # 3. Forward pass to find predicted class if target_category not specified
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_class = outputs.max(1)
    if target_category is None:
        target_category = pred_class.item()  # Use predicted class

    # 4. Create GradCAM object
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))

    # 5. Generate CAM for the target category
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Take the first (and only) image in batch

    # 6. Convert PIL image to NumPy
    img_np = np.array(img_pil, dtype=np.uint8)

    # 7. Overlay the CAM on the image
    # Convert to BGR for show_cam_on_image, scale to [0,1]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) / 255.0
    visualization = show_cam_on_image(img_bgr, grayscale_cam, use_rgb=True)

    # 8. Save or display the result
    cv2.imwrite(output_path, visualization)
    print(f"Grad-CAM visualization saved to {output_path}")

# --------------------------------
# (4) Main Function
# --------------------------------
def main():
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Define data augmentations (transforms)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandAugment(num_ops=2, magnitude=9),  # Requires torchvision>=0.10
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Load data from ImageFolder
    data_dir = "../data"  # <-- CHANGE if needed
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)  # 20% validation
    train_size = dataset_size - val_size

    # 4. Split into train and validation subsets
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transforms to the val subset
    val_dataset.dataset.transform = val_transforms

    # 5. Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8),
        'val':   DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)
    }

    # Print classes
    class_names = full_dataset.classes
    print("Detected classes:", class_names)

    # 6. Initialize ResNet18 (pretrained)
    model = models.resnet18(pretrained=True)
    
    # -------------------------------
    # Fine-Tuning Strategy:
    # Freeze layers except the last block (layer4) + final FC
    # -------------------------------
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace the final layer to match number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))

    model = model.to(device)

    # 7. Define loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 8. Define optimizer (lower LR for fine-tuning)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # 9. Define CosineAnnealingLR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    # 10. Train the model with Mixup
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=50,         # Increase if needed
        mixup_alpha=0.2        # Adjust mixup parameter
    )

    # 11. Save the best model weights
    os.makedirs("../models", exist_ok=True)
    model_save_path = "../models/aerial_activity_detector_robust.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # ----------------------------------------
    # (Optional) Example of Grad-CAM usage:
    # ----------------------------------------
    # Provide a path to a test image containing a helicopter (or any aerial activity)
    test_image_path = "../data/helicopter/000001.jpg"  # Change to an actual image path
    if os.path.exists(test_image_path):
        visualize_gradcam(
            model=model,
            device=device,
            image_path=test_image_path,
            transform=val_transforms,
            target_layer=model.layer4[-1],  # Last block of layer4 in ResNet18
            target_category=None,           # Use predicted class
            output_path="gradcam_helicopter.jpg"
        )

if __name__ == "__main__":
    main()
