import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import urllib
from typing import Any, Callable, Dict, List, Tuple
import PIL
from PIL import Image, ImageDraw, ImageFont
import spacy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import clip

FONT_SIZE = 13*2
COLOR_LIST = ["red", "green", "blue", "cyan", "yellow", "purple",
              "deeppink", "ghostwhite", "darkcyan", "olive",
              "orange", "orangered", "darkgreen"]

sam_checkpoint = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "SAM-BLIP2/sam_vit_h_4b8939.pth"
CHECKPOINT_URL = 'SAM-BLIP2/sam_vit_h_4b8939.pth'

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def nms(masks, iou_thresh):
    all_bbox = []
    reserved_bbox = []
    scores = []
    for mask in masks:
        x, y, w, h = mask["bbox"]
        score = mask['predicted_iou']
        all_bbox.append([x, y, x+w, y+h])
        scores.append(score)

    bboxes = np.array(all_bbox)
    scores = np.array(scores)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]  
    while index.size > 0:
        i = index[0]
        result.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores, result # index


def draw_rect(img, bbox, text, color):
    #h,w,c = img.shape
    font = cv2.FONT_HERSHEY_PLAIN  # for text
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = xmin + int(bbox[2])
    ymax = ymin + int(bbox[3])
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 3)
    cv2.putText(img, text, (xmin + 2, ymin+int(FONT_SIZE)), font, 2, color, 2)
    

nlp = spacy.load('en_core_web_sm')
def get_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append(chunk.text)
    return noun_phrases


def filter_masks(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float,
    stability_score_threshold: float,
    query: str,
    clip_threshold: float,
) -> List[Dict[str, Any]]:
    cropped_masks: List[PIL.Image.Image] = []
    filtered_masks: List[Dict[str, Any]] = []

    for mask in masks:
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
        ):
            continue
        filtered_masks.append(mask)
        cropped_masks.append(crop_image(image, mask))

    if query and filtered_masks:
        scores = get_scores(cropped_masks, query)
        filtered_masks = [
            filtered_masks[i]
            for i, score in enumerate(scores)
            if score > clip_threshold
        ]

    return filtered_masks


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
#     masked = image * np.expand_dims(mask["segmentation"], -1)
#     crop = masked[y : y + h, x : x + w]
    crop = image[y : y + h, x : x + w]
#     if h > w:
#         top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
#     else:
#         top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
#     # padding
#     crop = cv2.copyMakeBorder(
#         crop,
#         top,
#         bottom,
#         left,
#         right,
#         cv2.BORDER_CONSTANT,
#         value=(0, 0, 0),
#     )
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#     cv2.imwrite('/home/ssd5/haojing/SAM-BLIP2/data_visual/a.jpg', crop)
    
    crop = PIL.Image.fromarray(crop)
    return crop


def filter_contained_bbox(masks):
    all_bbox = []
    reserved_bbox = []
    scores = []
    for mask in masks:
        x, y, w, h = mask["bbox"]
        score = mask['predicted_iou']
        all_bbox.append([x, y, x+w, y+h])
        scores.append(score)

    bboxes = np.array(all_bbox)
    scores = np.array(scores)
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)
    
    
    
    result = []
    index = scores.argsort()[::-1]  
    while index.size > 0:
        i = index[0]
#         result.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        
        max_bbox_area = areas[i] + areas[index[1:]] - overlaps
        idx = np.where(max_bbox_area == areas[i])[0]
        
#         ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        idx = np.where(ious <= iou_thresh)[0]
        
        index = index[idx + 1]  
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores


def takeArea(elem):
    return elem['area']

def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image

def predict_step(i_image, device, gen_kwargs, feature_extractor, tokenizer, model):
    image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
    images = [i_image]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def load_mask_generator() -> SamAutomaticMaskGenerator:
    if not os.path.exists(os.path.join(CHECKPOINT_PATH)):
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device)
#     mask_generator = SamAutomaticMaskGenerator(sam)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=10,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1024,  # Requires open-cv to run post-processing
        )
    
    return mask_generator


# CLIP final filter
def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    
    return model, preprocess

def get_scores(crops: List[PIL.Image.Image], query: str, model, preprocess) -> torch.Tensor:
#     model, preprocess = load_clip()
    preprocessed = [preprocess(crop) for crop in crops]
    preprocessed = torch.stack(preprocessed).to(device)
    token = clip.tokenize(query).to(device)
    img_features = model.encode_image(preprocessed)
    txt_features = model.encode_text(token)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    txt_features /= txt_features.norm(dim=-1, keepdim=True)
    similarity = (100 * img_features @ txt_features.T).softmax(0)
    return similarity
