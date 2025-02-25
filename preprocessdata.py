import os
import cv2

def annotate_images_by_directory(
    data_dir: str,
    output_dir: str
):
    """
    For each subfolder in 'data_dir', treats the subfolder's name as the annotation label.
    Draws a bounding box around the entire image and writes the label on it.
    Saves the annotated images in 'output_dir' with the same subfolder structure.
    
    Parameters:
    -----------
    data_dir : str
        Path to the main dataset directory containing subfolders of images.
    output_dir : str
        Path to the directory where annotated images will be saved.
    """

    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Loop over each subfolder (each class/label)
    for label_name in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_name)
        
        # Skip if it's not a directory
        if not os.path.isdir(label_path):
            continue
        
        # Create the corresponding output subfolder
        out_label_dir = os.path.join(output_dir, label_name)
        os.makedirs(out_label_dir, exist_ok=True)
        
        # 3. For each image in the subfolder
        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(label_path, filename)
                
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load image: {img_path}")
                    continue
                
                # Dimensions of the image
                height, width, _ = img.shape
                
                # 4. Draw a bounding box around the entire image
                # Top-left corner (0, 0), bottom-right corner (width-1, height-1)
                box_color = (0, 255, 0)  # green in BGR
                cv2.rectangle(img, (0, 0), (width-1, height-1), box_color, 2)
                
                # 5. Put text (the directory name) on the top-left
                label_text = label_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img, label_text,
                    (10, 30),  # slightly down from top-left
                    font, 1.0,  # font scale
                    box_color,
                    2  # thickness
                )
                
                # 6. Save the annotated image
                out_img_path = os.path.join(out_label_dir, filename)
                cv2.imwrite(out_img_path, img)
                print(f"Annotated image saved -> {out_img_path}")

if __name__ == "__main__":
    # Example usage:
    DATA_DIR = "./data"               # e.g., "data/" with subfolders: "cat/", "dog/"
    OUTPUT_DIR = "./preprocessed_data" # output folder
    
    annotate_images_by_directory(DATA_DIR, OUTPUT_DIR)
