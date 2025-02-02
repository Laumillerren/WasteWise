import numpy as np
import torch
import torchvision.transforms as T
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_file(file_path, transform, return_type='np'):
    im_id = file_path.split('/')[-1]
    cat = file_path.split('/')[-2]
    #out_path = f"/home/luke/Drexel/Winter-25/project/tensors/{cat}" # Change path. output 
    out_path = f"/Users/laurenmiller/Documents/Drexel MSDS/Winter 24-25/Winter-25/project/tensors/{cat}" # Change path. output 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img = Image.open(file_path)
    img_tensor = transform(img)
    if return_type == 'np':
        img_tensor = img_tensor.numpy()
        np.save(f"{out_path}/{im_id}.npy", img_tensor)
    elif return_type == 'pt':
        torch.save(img_tensor, f"{out_path}/{im_id}.pt")

if __name__ == '__main__':
    #dir_path = '/home/luke/Drexel/Winter-25/project/data_minips'# Change path. output from im_manipulation
    dir_path = '/Users/laurenmiller/Documents/Drexel MSDS/Winter 24-25/Winter-25/project/data_minips'# Change path. output from im_manipulation
    all_files = [os.path.join(root, f)
                 for root, _, files in os.walk(dir_path)
                 for f in files]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(process_file, file, transform) for file in all_files]
        for _ in tqdm(as_completed(futures), total=len(all_files)):
            pass





