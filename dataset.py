from torch.utils import data
import parameters as p
from torch import float32
from torchvision import datasets,transforms

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(p.IMAGE_CROP_SIZE,transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(p.IMAGE_CROP_SIZE),
    transforms.ConvertImageDtype(float32)])

# dataset = datasets.mnist.MNIST('X:\Machine_Learning\dataset\\',download=True)

def get_dataloader(presistent_workers=True,num_workers=4):

    dataset = datasets.ImageFolder(p.DATASET_PATH,transform=train_transforms)
    
    return data.DataLoader(dataset,batch_size=p.BATCH_SIZE,persistent_workers=presistent_workers,num_workers=num_workers)


if __name__ == "__main__":
    import warnings
    import cv2
    import einops
    from torch import Tensor
    warnings.filterwarnings("ignore", category=UserWarning)
    cv2.namedWindow('Sample',cv2.WINDOW_AUTOSIZE)
    for sample,_ in get_dataloader():
        sample:Tensor
        sample = einops.rearrange(
                    sample, '(b_x b_y) c h w -> (b_y h) (b_x w) c', b_x=2).numpy()
        # img = einops.rearrange(sample,'1 c h w -> h w c').numpy()
        img = cv2.cvtColor(sample,cv2.COLOR_RGB2BGR)
        # print(img)
        cv2.imshow("Sample",img)
        cv2.waitKey(0)

    # datasetloader = get_dataloader()
    # dataset = datasets.ImageFolder(p.DATASET_PATH,transform=train_transforms)
    # print(len(dataset))

