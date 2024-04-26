from os import listdir
from os.path import join
from pathlib import Path
import numpy as np
import shutil

def extract_file_id(filename: str) -> str:
    """
    Extract the unique file ID from the filename based on the format '{id}__{other_info}'.
    """
    return filename.split('__')[0]

# function that given all the file ids, extract the unique 
def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        print(x)
    return

def get_random_images(img_dir: Path, mask_dir: Path, n_samples=1000, seed=None, filter_ids=None):
    # Get a list of all the images
    all_images = [file for file in img_dir.iterdir() if file.is_file()]
    
    # Extract unique file ids and filter them
    file_ids = [extract_file_id(file.name) for file in all_images]
    # unique(file_ids)
    if filter_ids:
        # Filter file IDs based on a condition or a predefined list
        filtered_images = [file for file, file_id in zip(all_images, file_ids) if file_id in filter_ids]
    else:
        filtered_images = all_images

    # Take a random sample of images
    if seed:
        np.random.seed(seed)
    if len(filtered_images) < n_samples:
        raise ValueError("Not enough files after filtering to sample the requested number of images.")
    sample = list(np.random.choice(filtered_images, n_samples, replace=False))

    # Extract the corresponding masks
    masks = [mask_dir / file.name for file in sample]
    return sample, masks

# function to move the images and masks to the data folder
def move_files(files: list, dest: Path):
    # Ensure the destination directory exists, if not, create it
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Copy the file
    try:
        for file in files:
            shutil.copy(file, dest)
    except Exception as e:
        return e
    

if __name__ == '__main__':
    dir_img = Path("C:/Users/steel/Downloads/dataset/pathtools_gleason_grading/images/")
    dir_mask = Path("C:/Users/steel/Downloads/dataset/pathtools_gleason_grading/masks/")
    dest_img = Path("C:/Users/steel/Downloads/Pytorch-UNet/data/imgs/")
    dest_mask = Path("C:/Users/steel/Downloads/Pytorch-UNet/data/masks/")
    filter_ids = ['772233_1625116306','739526_1622166900','788846_1626753261']

    imgs, masks = get_random_images(dir_img, dir_mask, 15000, filter_ids=filter_ids)
    moved = move_files(imgs, dest_img)
    move = move_files(masks, dest_mask)

# def get_random_images(img_dir: Path, mask_dir: Path, n_samples=1000, seed=None):
#     # get a list of all the images
#     all_images = [file for file in img_dir.iterdir()]
#     # get unique file ids

#     # take random sample of images
#     if seed:
#         np.random.seed(seed)
#     sample = list(np.random.choice(all_images, n_samples, replace=False))
#     # extract the corresponding mask
#     masks = [mask_dir / file.name for file in sample]
#     return sample, masks
