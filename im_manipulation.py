import os
from PIL import Image, ImageOps, ImageEnhance,ImageFilter
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def noise(img, noise_level=50):
    img = np.array(img)
    noise = np.random.randint(0,noise_level, img.shape,dtype='uint8')
    noise = np.clip(img + noise,0,255)

    return Image.fromarray(noise.astype('uint8'))

def augment_image(path):
    img = Image.open(path)

    aug = {}

    aug['original'] = img

    aug['greyscale'] = ImageOps.grayscale(img)
    aug['rot_90'] = img.rotate(90, expand = True)
    aug['rot_180'] = img.rotate(180, expand = True)
    aug['rot_270'] = img.rotate(270, expand = True)

    aug['h_flip'] = ImageOps.mirror(img)
    aug['v_filp'] = ImageOps.flip(img)

    aug['noise'] = noise(img)

    e = ImageEnhance.Brightness(img)

    aug['bright'] = e.enhance(1.5)
    aug['dark'] = e.enhance(0.5)

    aug['blur'] = img.filter(ImageFilter.BLUR)

    if img.mode != 'RGB':
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img

    aug['invert'] = ImageOps.invert(img_rgb)

    aug['posterize'] = ImageOps.posterize(img, 4)
    aug['solarize'] = ImageOps.solarize(img,128)

    aug['equalize'] = ImageOps.equalize(img)

    

    return aug


def process_file(file):
    im_id = file.split('/')[-1]
    cat = file.split('/')[-2]
    #out_path = f"/home/luke/Drexel/Winter-25/project/data_minips/{cat}"# Change path. where the data is going 
    out_path = f"/Users/laurenmiller/Documents/Drexel MSDS/Winter 24-25/Winter-25/project/data_minips/{cat}"# Change path. where the data is going 
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    manipulations = augment_image(file)
    for name, im in manipulations.items():
        im.save(f'{out_path}/{im_id}_{name}.JPG')
    return file

if __name__ == '__main__':
    #dir_path = '/home/luke/Drexel/Winter-25/project/waste_dataset'# Change path. where data is stored
    dir_path = '/Users/laurenmiller/Documents/Drexel MSDS/Winter 24-25/Winter-25/project/waste_dataset'# Change path. where data is stored
    all_files = [os.path.join(root, f)
                 for root, _, files in os.walk(dir_path)
                 for f in files]

    with ThreadPoolExecutor(max_workers=28) as executor:
        futures = [executor.submit(process_file, file) for file in all_files]
        for _ in tqdm(as_completed(futures), total=len(all_files)):
            pass
