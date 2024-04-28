from fastai.vision.all import *
from fastai.losses import *
from fastai.callback.wandb import WandbCallback
import wandb

from unet.resnet_unet import UNetTrained
from utils.dice_score import dice_loss

wandb.init(project='U-Net', resume='allow', anonymous='must', 
           name='unet-10000-10-aug', notes='10000 samples, 10 epochs, applied augmentations')

def get_subset_files(path, n_samples=100, seed = 42):
    # Get all image files
    all_files = get_image_files(path)
    # Randomly sample from the list of files
    random.seed(seed)  # For reproducibility
    selected_files = random.sample(all_files, n_samples)
    return selected_files
            

path = Path('C:/Users/steel/Downloads/Pytorch-UNet/data')
codes = np.loadtxt(path/'codes.txt', dtype=str)  # if you have class codes for segmentation

params = {
    'seed': 50,
    'n_samples':10000,
    'epochs': 10,
}
# batch_tfms = aug_transforms(do_flip=True, flip_vert=False, max_rotate=30.0, min_zoom=1.0, max_zoom=2.0, max_lighting=0.3, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
datablock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                      get_items=partial(get_subset_files, n_samples=params.get('n_samples'), seed=params.get('seed')),
                      splitter=RandomSplitter(valid_pct=0.2, seed=params.get('seed')),
                      get_y=lambda o: path/'masks'/f'{o.stem}{o.suffix}',
                      batch_tfms=[*aug_transforms(do_flip=True, flip_vert=False, max_rotate=30.0, min_zoom=1.0, max_zoom=2.0, max_lighting=0.3, max_warp=0.2, p_affine=0.75, p_lighting=0.75)],
                    #   num_workers = 0
                      )
dls = datablock.dataloaders(path/'imgs', bs=3, num_workers=0)
dls.show_batch()
# plt.show()

learn = Learner(dls, UNetTrained(in_channels=3, out_channels=5), loss_func=dice_loss, opt_func=Adam, cbs=[WandbCallback(log_model=False)])

# learn = unet_learner(dls, resnet18, pretrained=False, metrics=[DiceMulti])
# learn.model = learn.model.g
learn.fit_one_cycle(params.get('epochs'), 1e-5,cbs=[ReduceLROnPlateau(monitor='valid_loss', patience=2), GradientClip(1)])
# learn.show_results()
# plt.show()