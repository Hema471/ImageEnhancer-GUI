import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def enhance_image(input_image_path, output_image_path, model_path="EDSR_x2.pb", scale_factor=2):
    # Check if the input image path exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Check if the model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {input_image_path}")
    
    # Load the pre-trained model 
    model = cv2.dnn_superres.DnnSuperResImpl_create()
    model.readModel(model_path)
    model.setModel("edsr", scale_factor)
    
    # Upscale the image to enhance quality
    enhanced_image = model.upsample(image)
    
    # Save the enhanced image
    cv2.imwrite(output_image_path, enhanced_image)
    print(f"Enhanced image saved to {output_image_path}")
    
    # Display the images after enhancement
    cv2.imshow("Input Image", image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        input_path_var.set(file_path)

def select_output_path():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("Bitmap files", "*.bmp")])
    if file_path:
        output_path_var.set(file_path)

def process_image():
    try:
        input_image_path = input_path_var.get()
        output_image_path = output_path_var.get()
        model_choice = model_var.get()
        
        model_map = {
            "EDSR_x2": ("EDSR_x2.pb", 2),
            "EDSR_x3": ("EDSR_x3.pb", 3),
            "EDSR_x4": ("EDSR_x4.pb", 4)
        }
        
        model_path, scale_factor = model_map[model_choice]
        
        enhance_image(input_image_path, output_image_path, model_path, scale_factor)
        messagebox.showinfo("Success", "Image enhanced and saved successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI 
root = tk.Tk()
root.title("Image Enhancer")

tk.Label(root, text="Input Image Path:").grid(row=0, column=0, padx=10, pady=5)
input_path_var = tk.StringVar()
tk.Entry(root, textvariable=input_path_var, width=50).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_image).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Output Image Path:").grid(row=1, column=0, padx=10, pady=5)
output_path_var = tk.StringVar()
tk.Entry(root, textvariable=output_path_var, width=50).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_output_path).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="Choose Model:").grid(row=2, column=0, padx=10, pady=5)
model_var = tk.StringVar(value="EDSR_x2")
tk.OptionMenu(root, model_var, "EDSR_x2", "EDSR_x3", "EDSR_x4").grid(row=2, column=1, padx=10, pady=5)

tk.Button(root, text="Enhance Image", command=process_image).grid(row=3, column=0, columnspan=3, pady=10)

root.mainloop()
