import cv2
import numpy as np
import os  # 引入os模块


from skimage.transform import resize

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

from collections import deque

def select_points(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN and len(params[0]) < 4:
        params[0].append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if len(params[0]) > 1:
            cv2.line(frame, params[0][-1], params[0][-2], (255, 0, 0), 2)


def estimate_missing_point(pts):
    if len(pts) != 3:
        return None
    p1, p2, p3 = pts[0], pts[1], pts[2]
    p4 = p1 + p3 - p2
    return p4

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_rect_size(points):
    width = (calculate_distance(points[0], points[1]) + calculate_distance(points[2], points[3])) / 2
    height = (calculate_distance(points[0], points[3]) + calculate_distance(points[1], points[2])) / 2
    return int(width), int(height)

def sort_points(points):
    center = points.mean(axis=0)
    angles = np.arctan2(points[:,1] - center[1], points[:,0] - center[0])
    return points[np.argsort(angles)]

class Intrinsic():
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
intrinsic = Intrinsic(_paper=False)
video_path = 'test9.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()


fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


output_path = 'tracked_video_9.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)


cv2.imshow('Select 4 Points', frame)
points = []
cv2.setMouseCallback('Select 4 Points', select_points, [points])
while len(points) < 4:
    cv2.imshow('Select 4 Points', frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()

points = np.float32(points)
points = sort_points(points)
initial_points = points.copy()
rect_size = calculate_rect_size(points)
poster = cv2.imread('poster.png')
poster = cv2.resize(poster, rect_size)

poster_points = np.float32([[0, 0], [rect_size[0], 0], [rect_size[0], rect_size[1]], [0, rect_size[1]]])

# 追
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame_queue = deque(maxlen=5)
brightness_queue = deque(maxlen=5)

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    beta = brightness - np.mean(img)
    alpha = 1 + contrast / 127.0  # 假设对比度从-127到+127变化
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)

    good_points = []
    for i, (new, good) in enumerate(zip(new_points, status.flatten())):
        if good:
            good_points.append(new)
        else:
            # 估算缺失点
            remaining_points = [good_points[i] for i in range(len(good_points)) if i != i]
            estimated = estimate_missing_point(remaining_points)
            if estimated is not None:
                good_points.append(estimated)
            else:
                good_points.append(points[i])  # 用上一个位置代替
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, points, None, **lk_params)
    good_points = np.array(good_points, dtype=np.float32)

    M = cv2.getPerspectiveTransform(poster_points, good_points)
    warped_poster = cv2.warpPerspective(poster, M, (frame.shape[1], frame.shape[0]))

    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(good_points), (255,) * frame.shape[2])
    frame = intrinsic.run(
            bg_img=frame,
            fg_img=warped_poster,
            mask_img=mask,
            good_points=good_points
        )
    if len(frame_queue) < 5:
        frame_queue.append(frame)
        # 计算并存储当前帧的局部亮度
        local_brightness = np.mean(frame[mask])
        brightness_queue.append(local_brightness)
    else:
        frame_queue.popleft()
        frame_queue.append(frame)
        brightness_queue.popleft()
        local_brightness = np.mean(frame[mask])
        brightness_queue.append(local_brightness)

        # 计算亮度和对比度的平均值
        mean_brightness = np.mean(brightness_queue)
        brightness_adjustment = mean_brightness - local_brightness

        # 调整当前帧的局部亮度和对比度
        frame_adjusted = adjust_brightness_contrast(frame[mask], brightness=brightness_adjustment)
        frame[mask] = frame_adjusted
    previous_frame = frame.copy()
    cv2.imshow('Tracked', frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    old_gray = gray_frame.copy()
    points = good_points

cap.release()
out.release()
cv2.destroyAllWindows()