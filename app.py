import gradio as gr
import glob
import torch
import joblib
from PIL import Image, ImageDraw
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from class_info import class_names, class_colors, class_ids
import numpy as np
from scipy.ndimage import center_of_mass


def combine_ims(im1, im2, val=128):
  p = Image.new("L", im1.size, val)
  im = Image.composite(im1, im2, p)
  return im

def get_class_centers(segmentation_mask, class_dict):
    segmentation_mask = segmentation_mask.numpy() + 1
    class_centers = {}
    for class_index, _ in class_dict.items():
        class_mask = (segmentation_mask == class_index).astype(int)
        center_of_mass_list = center_of_mass(class_mask)
        
        class_centers[class_index] = center_of_mass_list
    
    class_centers = {k:list(map(int, v)) for k,v in class_centers.items() if not np.isnan(sum(v))}
    return class_centers

def visualize_mask(predicted_semantic_map, class_ids, class_colors):
  h, w = predicted_semantic_map.shape
  color_indexes = np.zeros((h, w), dtype=np.uint8)
  color_indexes[:] = predicted_semantic_map.numpy()
  color_indexes = color_indexes.flatten()

  colors = class_colors[class_ids[color_indexes]]
  output = colors.reshape(h, w, 3).astype(np.uint8)
  image_mask = Image.fromarray(output)
  return image_mask


def get_out_image(image, predicted_semantic_map):
  class_centers = get_class_centers(predicted_semantic_map, class_dict) 
  mask = visualize_mask(predicted_semantic_map, class_ids, class_colors)
  image_mask = combine_ims(image, mask, val=128)
  draw = ImageDraw.Draw(image_mask)
  for id, (y, x) in class_centers.items():
    draw.text((x, y), str(class_names[id-1]), fill='black')

  return image_mask

def gradio_process(image):
  inputs = processor(images=image, return_tensors="pt")

  with torch.no_grad():
      outputs = model(**inputs)

  predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

  out_image = get_out_image(image, predicted_semantic_map)
  return out_image

#class_names, class_ids, class_colors = joblib.load('ade20k_classes.joblib')
class_names, class_ids, class_colors = np.array(class_names), np.array(class_ids), np.array(class_colors)
class_dict = dict(zip(class_ids, class_names))

device = torch.device("cpu")
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic").to(device)
model.eval()

demo = gr.Interface(
    gradio_process, 
    inputs=gr.inputs.Image(type="pil"), 
    outputs=gr.outputs.Image(type="pil"),
    title="Semantic Interior Segmentation",
    examples=glob.glob('./examples/*.jpg'),
    allow_flagging="never",

)

demo.launch()