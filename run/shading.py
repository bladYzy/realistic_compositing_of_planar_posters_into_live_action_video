import cv2
import numpy as np

from chrislib.general import invert, uninvert, view, np_to_pil, to2np, add_chan
from chrislib.data_util import load_image
from chrislib.normal_util import get_omni_normals

from boosted_depth.depth_util import create_depth_models, get_depth

from intrinsic.model_util import load_models
from intrinsic.pipeline import run_pipeline

from intrinsic_compositing.shading.pipeline import (
    load_reshading_model,
    compute_reshading,
    generate_shd,
    get_light_coeffs
)

from intrinsic_compositing.albedo.pipeline import (
    load_albedo_harmonizer,
    harmonize_albedo
)

from omnidata_tools.model_util import load_omni_model


class Shading():
    def __init__(self, _paper=False):
        self.paper = _paper
        print('loading depth model')
        self.dpt_model = create_depth_models()

        print('loading normals model')
        self.nrm_model = load_omni_model()

        print('loading intrinsic decomposition model')
        self.int_model = load_models('paper_weights')

        print('loading albedo model')
        self.alb_model = load_albedo_harmonizer()

        print('loading reshading model')
        if self.paper:
            self.shd_model = load_reshading_model('paper_weights')
        else:
            self.shd_model = load_reshading_model('further_trained')
    
    def run(self, bg_img, fg_img, mask_img, good_points):
        # compute normals and shading for background, and use them 
        # to optimize for the lighting coefficients
        bg_img = self.convert_opencv_image_to_format(bg_img)
        fg_img = self.convert_opencv_image_to_format(fg_img)
        mask_img = self.convert_opencv_image_to_format(mask_img)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, :1]
        else:
            mask_img = mask_img[:, :, np.newaxis]
        MAX_BG_SZ = 1024
        bg_h, bg_w, _ = bg_img.shape
        
        # resize the background image to be large side < some max value
        max_dim = max(bg_h, bg_w)
        scale = MAX_BG_SZ / max_dim
        bg_nrm = get_omni_normals(self.nrm_model, bg_img)
        result = run_pipeline(
            self.int_model,
            bg_img ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )
        bg_shd = result['inv_shading'][:, :, None]

        max_dim = max(bg_h, bg_w)
        scale = 512 / max_dim
        small_bg_img = self.rescale(bg_img, scale)
        small_bg_nrm = get_omni_normals(self.nrm_model, small_bg_img)
        small_bg_shd = self.rescale(bg_shd, scale)

        orig_coeffs, _ = get_light_coeffs(
            small_bg_shd[:, :, 0], 
            small_bg_nrm, 
            small_bg_img
        )

        coeffs = orig_coeffs
        bb = self.find_bounds(good_points)

        
        bb_h, bb_w = bb[1] - bb[0], bb[3] - bb[2]

        loc_y = bg_h // 2
        loc_x = bg_w // 2
        orig_fg_crop = fg_img[bb[0] : bb[1], bb[2] : bb[3], :].copy()
        orig_msk_crop = mask_img[bb[0] : bb[1], bb[2] : bb[3], :].copy()

        max_dim = max(bb_h, bb_w)

        real_scale = MAX_BG_SZ / max_dim

        orig_fg_crop = self.rescale(orig_fg_crop, real_scale)
        orig_msk_crop = self.rescale(orig_msk_crop, real_scale)

        max_dim = max(orig_fg_crop.shape)
        disp_scale = (min(bg_h, bg_w) // 2) / max_dim
        frag_scale = disp_scale
        print('init frag_scale:', frag_scale)

        fg_crop = self.rescale(orig_fg_crop, frag_scale)
        msk_crop = self.rescale(orig_msk_crop, frag_scale)

        bb_h, bb_w, _ = fg_crop.shape

        orig_fg_nrm = get_omni_normals(self.nrm_model, orig_fg_crop)

        result = run_pipeline(
            self.int_model,
            orig_fg_crop ** 2.2,
            resize_conf=0.0,
            maintain_size=True,
            linear=True
        )

        orig_fg_shd = result['inv_shading'][:, :, None]

        bg_dpt = get_depth(bg_img, self.dpt_model)[:, :, None]
        orig_fg_dpt = get_depth(orig_fg_crop, self.dpt_model)[:, :, None]

        fg_shd_res = self.rescale(orig_fg_shd, frag_scale)
        fg_nrm_res = self.rescale(orig_fg_nrm, frag_scale)
        fg_dpt_res = self.rescale(orig_fg_dpt, frag_scale)
        
        top = bb[0]
        left = bb[2]

        comp_img = self.composite_crop(
            bg_img, 
            (top, left),
            fg_crop,
            msk_crop
        )

        comp_shd = self.composite_crop(
            bg_shd, 
            (top, left),
            fg_shd_res,
            msk_crop
        )

        comp_msk = self.composite_crop(
            np.zeros_like(bg_shd), 
            (top, left),
            msk_crop,
            msk_crop
        )

        comp_nrm = self.composite_crop(
            bg_nrm,
            (top, left),
            fg_nrm_res,
            msk_crop
        )

        comp_dpt = self.composite_depth(
            bg_dpt,
            (top, left),
            fg_dpt_res,
            msk_crop
        )
        
        # the albedo comes out gamma corrected so make it linear
        alb_harm = harmonize_albedo(
            comp_img, 
            comp_msk, 
            comp_shd, 
            self.alb_model,
            reproduce_paper=self.paper
        ) ** 2.2

        harm_img = alb_harm * uninvert(comp_shd)

        result = compute_reshading(
            harm_img,
            comp_msk,
            comp_shd,
            comp_dpt,
            comp_nrm,
            alb_harm,
            coeffs,
            self.shd_model
        )

        return self.convert_format_to_opencv_image(result['composite'])


    def rescale(self, img, scale):
        # if scale == 1.0: return img

        # h = img.shape[0]
        # w = img.shape[1]

        # img = resize(img, (int(h * scale), int(w * scale)))
        return img

    def find_bounds(self, points):
        rmin = float('inf')
        rmax = float('-inf')
        cmin = float('inf')
        cmax = float('-inf')
        for r, c in points:
            if r < rmin:
                rmin = r
            if r > rmax:
                rmax = r
            if c < cmin:
                cmin = c
            if c > cmax:
                cmax = c
        return int(cmin), int(cmax), int(rmin), int(rmax)

    def composite_crop(self, img, loc, fg, mask):
        c_h, c_w, _ = fg.shape 

        img = img.copy()
        
        img_crop = img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :]
        comp = (img_crop * (1.0 - mask)) + (fg * mask)
        img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w, :] = comp

        return img

    def composite_depth(self, img, loc, fg, mask):
        c_h, c_w = fg.shape[:2]

        # get the bottom-center depth of the bg
        bg_bc = loc[0] + c_h, loc[1] + (c_w // 2)
        bg_bc_val = img[bg_bc[0], bg_bc[1]].item()

        # get the bottom center depth of the fragment
        fg_bc = c_h - 1, (c_w // 2)
        fg_bc_val = fg[fg_bc[0], fg_bc[1]].item()

        # compute scale to match the fg values to bg
        scale = bg_bc_val / fg_bc_val

        img = img.copy()
        
        img_crop = img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w]
        comp = (img_crop * (1.0 - mask)) + (scale * fg * mask)
        img[loc[0] : loc[0] + c_h, loc[1] : loc[1] + c_w] = comp

        return img
    def convert_opencv_image_to_format(self, image, bits=8):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32)
        image_normalized = image_float / float((2 ** bits) - 1)
        return image_normalized
    def convert_format_to_opencv_image(self, image, bits=8):
        max_val = (2 ** bits) - 1
        image_unnormalized = image * max_val
        image_uint8 = np.clip(image_unnormalized, 0, max_val).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
        
        return image_bgr




'''
@INPROCEEDINGS{careagaCompositing,  
	author={Chris Careaga and S. Mahdi H. Miangoleh and Ya\u{g}{\i}z Aksoy},  
	title={Intrinsic Harmonization for Illumination-Aware Compositing},  
	booktitle={Proc. SIGGRAPH Asia},  
	year={2023},  
}

@ARTICLE{careagaIntrinsic,
  author={Chris Careaga and Ya\u{g}{\i}z Aksoy},
  title={Intrinsic Image Decomposition via Ordinal Shading},
  journal={ACM Trans. Graph.},
  year={2023},
}

'''
