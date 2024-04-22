import torch
from torchvision import transforms
import PIL
from PIL import Image

import os.path
from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
from omnidata_tools.torch.data.transforms import get_transform

class GetNormal:
    def __init__(self, image_og, output_path):
        self.image_og = image_og
        self.image_width, self.image_height = self.image_og.size
        self.output_path = output_path
        self.root_dir = './pretrained_models/'
        self.map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_size = 384
        self.image= self.image_og.resize((self.image_size, self.image_size), Image.LANCZOS)
        self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        self.trans_totensor = transforms.Compose([transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
                                                  transforms.CenterCrop(self.image_size),
                                                  get_transform('rgb', image_size=None)])
        self.trans_topil = transforms.ToPILImage()

    def save_outputs(self):
        with torch.no_grad():
            save_path = os.path.join(self.output_path, 'normal_map.png')
            img_tensor = self.trans_totensor(self.image)[:3].unsqueeze(0).to(self.device)
            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)
            output = self.model(img_tensor).clamp(min=0, max=1)
            normal_map = self.trans_topil(output[0])
            resized_normal_map = normal_map.resize((self.image_width, self.image_height), Image.LANCZOS)

            resized_normal_map.save(save_path)
            print(f'Writing output {save_path} ...')
            '''
            self.trans_topil(output[0]).save(save_path)
            print(f'Writing output {save_path} ...')
            normal_map = self.trans_topil(output[0])
            normal_map=normal_map.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            '''
            return resized_normal_map

    def set_model(self):
        pretrained_weights_path = self.root_dir + 'omnidata_dpt_normal_v2.ckpt'
        checkpoint = torch.load(pretrained_weights_path, map_location=self.map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


    def run(self):
        self.set_model()
        return self.save_outputs()

