from fastai.vision.all import *
from fastai.losses import *

from unet.unet_model import UNet
from utils.dice_score import dice_loss

path = Path('C:/Users/steel/Downloads/Pytorch-UNet/data')
codes = np.loadtxt(path/'codes.txt', dtype=str)  # if you have class codes for segmentation

datablock = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                      get_items=get_image_files,
                      splitter=RandomSplitter(valid_pct=0.2, seed=42),
                      get_y=lambda o: path/'masks'/f'{o.stem}{o.suffix}',
                      batch_tfms=[*aug_transforms(size=(256,256))],
                    #   num_workers = 0
                      )
dls = datablock.dataloaders(path/'imgs', bs=3, num_workers=0)
dls.show_batch()
plt.show()

learn = Learner(dls, UNet(in_channels=3, out_channels=5), loss_func=dice_loss)

learn = unet_learner(dls, resnet18, pretrained=False, metrics=[DiceMulti])
# learn.model = learn.model.g
learn.fit_one_cycle(5)
learn.show_results(max_n=6, figsize=(7,8))