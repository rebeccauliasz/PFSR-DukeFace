from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from os.path import join
from PIL import Image
import glob


#I just ripped out the CelebA stuff, it does work though (check the github for the origial version of this file for that)
class CelebDataSet(Dataset):
    """CelebA dataset
    Parameters:
        data_path (str)     -- CelebA dataset main directory(inculduing '/Img' and '/Anno') path
        state (str)         -- dataset phase 'train' | 'val' | 'test'

    Center crop the alingned celeb dataset to 178x178 to include the face area and then downsample to 128x128(Step3).
    In addition, for progressive training, the target image for each step is resized to 32x32(Step1) and 64x64(Step2).
    """

    def __init__(self, data_path = './dataset/', state = 'train'):
        self.main_path = data_path
        self.state = state

        #self.img_path = join(self.main_path, 'CelebA/Img/img_align_celeba')
        self.img_path = join(self.main_path, 'test-img')
        #self.eval_partition_path = join(self.main_path, 'Anno/list_eval_partition.txt')

        train_img_list = []
        val_img_list = []
        test_img_list = []

        image_list = []
        for filename in sorted(glob.glob('*.png')):
            #im=Image.open(filename)
            image_list.append(filename)

        for filename in sorted(glob.glob('*.jpg')):
            #im=Image.open(filename)
            image_list.append(filename)

        for filename in sorted(glob.glob('*.JPG')):
            #im=Image.open(filename)
            image_list.append(filename)

        self.image_list = image_list


        #image center cropping
        #Let's just use whatever we get
        self.pre_process = transforms.Compose([
                                    #These steps are used when training on Celeba dataset
                                    #transforms.Resize((178, 178)),
                                    #transforms.CenterCrop((178, 178)),
                                    transforms.Resize((128, 128)),
                                    ])

        self.totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

        self._64x64_down_sampling = transforms.Resize((64, 64))
        self._32x32_down_sampling = transforms.Resize((32, 32))
        self._16x16_down_sampling = transforms.Resize((16,16))

    #You can modify this processing chain if your inputs are 16x16, though I actualy find pixel art tends to upscale better if with it in place
    #I usually tried various options and picked the best
    #(resized UP and then back down blurns the pixels, allows for more model creativity it seems...)
    #I also iteratively ran inputs back in, which was really effective with emoji and pixel sprites who usually grew eyes in 2 or 3 iterations, but that code is a mess and not in this notebook
    def __getitem__(self, index):

        image_path = join(self.img_path, self.image_list[index])
        target_image = Image.open(image_path).convert('RGB')
        target_image = self.pre_process(target_image)
        x4_target_image = self._64x64_down_sampling(target_image)
        x2_target_image = self._32x32_down_sampling(x4_target_image)
        input_image = self._16x16_down_sampling(x2_target_image)

        x2_target_image = self.totensor(x2_target_image)
        x4_target_image = self.totensor(x4_target_image)
        target_image = self.totensor(target_image)
        input_image = self.totensor(input_image)


        return x2_target_image, x4_target_image, target_image, input_image


    def __len__(self):
        return len(self.image_list)
