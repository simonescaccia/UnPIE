import matplotlib.pyplot as plt
import cv2
from PIL import Image

from utils.pie_utils import jitter_bbox, squarify

num_dashes = 100

def write_dict(d, filename):
    with open(filename, 'w') as f:
        _write_dict(d, f)
        
def _write_dict(d, f, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            f.write('  '*indent + f"{k}:\n")
            _write_dict(v, f, indent+1)
        else:
            f.write('  '*indent + f"{k}: {v}\n")

def print_separator(message=None, space=True, top_new_line=True, bottom_new_line=True):
    if space and top_new_line:
        print()
    print('-'*num_dashes)
    if message:
        print(message)
    if space and bottom_new_line:
        print()
    return

def print_model_size(model, model_name):
    # Calculate the number of parameters
    num_params = model.count_params()
    # Assuming each parameter is 4 bytes (32 bits)
    memory_bytes = num_params * 4
    # Convert to megabytes
    memory_mb = memory_bytes / (1024 ** 2)
    print('')
    print(model_name, f" memory: {memory_mb:.2f} MB\n")

def print_image_with_bbox(image, df):
    # Convert CV image to PIL image
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image1)
    
    for i, row in df.iterrows():
        image_i = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        b = list(row['bbox'])
        bbox = jitter_bbox([b],'enlarge', 2, image=image_pil)[0]
        bbox = squarify(bbox, 1, image_pil.size[0])
        bbox = list(map(int,bbox[0:4]))
        cv2.rectangle(image_i, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        plt.imshow(image_i)
        plt.axis('off')
        plt.show()

    plt.imshow(image2)
    plt.axis('off')
    plt.show()
