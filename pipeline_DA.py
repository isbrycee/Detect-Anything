import os
import argparse
from utils import *
import torch
import torch.nn.functional as F
import sys
from random import randint
sys.path.append("..")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Detect Anything.')
    parser.add_argument('--data_dir', default='/home/ssd5/haojing/SAM-BLIP2/coco_100_data/', help='specify the root path of images and masks')
    parser.add_argument('--out_dir', default='/home/ssd5/haojing/SAM-BLIP2/data_visual/', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    args = parser.parse_args()
    return args
    
    
def main(rank, args):
    # img caption
    model_gpt2 = VisionEncoderDecoderModel.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/vit-gpt2-image-captioning").to(device)
    feature_extractor = ViTImageProcessor.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/vit-gpt2-image-captioning")
    tokenizer_gpt2 = AutoTokenizer.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/vit-gpt2-image-captioning")
    
    # VQA
    blip_processor = BlipProcessor.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/blip-vqa-base")
    blip_model = BlipForQuestionAnswering.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/blip-vqa-base").to(device)
    
    # flan-t5-base LLM Language-Aware Check
    tokenizer = T5Tokenizer.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("/home/ssd5/haojing/SAM-BLIP2/flan-t5-base", device_map="auto")
    
    data_dir = args.data_dir
    max_length = 32
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    nlp = spacy.load('en_core_web_sm')
    
    # VL-check
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    for file in os.listdir(data_dir):
        image = cv2.imread(os.path.join(data_dir, file))
        
        out1 = predict_step(image, device, gen_kwargs, feature_extractor, tokenizer_gpt2, model_gpt2)[0] # ['a woman in a hospital bed with a woman in a hospital bed']
        print('The image caption is ' + out1)

        # noun
        noun_list = get_noun_phrases(out1)

        print('The nouns of the image caption are: ' +  ", ".join(noun_list))

        # sam
        mask_generator = load_mask_generator()
        masks = mask_generator.generate(image)

        # nms
        iou_thresh = 0.9
        bboxes, scores, result = nms(masks, iou_thresh)
        masks = [masks[i] for i in result]

        print("There are %d bbox." % len(masks))

        # sort by area
        masks.sort(key=takeArea)


#         question = "This sub-image is cropped by the initial image which describes " + out1 + ", so what is this?"
        question = 'What is this object?'
        for mask in masks:
            if mask['area'] <=1024:
                continue
            sub_img = crop_image(image, mask)
            inputs = blip_processor(sub_img, question, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs)
            class_name = blip_processor.decode(out[0], skip_special_tokens=True)
            prompt = 'Question: ' + question + ". Answer: " + class_name + '.'
            print(prompt)

            check_flag = False
            for noun in noun_list:
                input_text = 'Does the ' + class_name + ' belong to the ' + noun + '?'
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(input_ids)
                flanT5_answer1 = tokenizer.decode(outputs[0])
                
                input_text = 'Is the ' + class_name + ' similiar to the ' + noun + '?'
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(input_ids)
                flanT5_answer2 = tokenizer.decode(outputs[0])
                
                if flanT5_answer1 == '<pad> yes</s>' or flanT5_answer2 == '<pad> yes</s>':
                    check_flag =True
                    continue
                print(input_text + " flan-T5: " + flanT5_answer2)

            if check_flag == False:
                continue
            else:
                print('Find a ' + class_name)
                
            #### CLIP check ####
                image_clip = clip_preprocess(sub_img).unsqueeze(0).to(device)                        
                text = clip.tokenize(["This is an image of %s" % class_name]).to(device)
                with torch.no_grad():
                    img_features = clip_model.encode_image(image_clip)
                    txt_features = clip_model.encode_text(text)
                    img_features /= img_features.norm(dim=-1, keepdim=True)
                    txt_features /= txt_features.norm(dim=-1, keepdim=True)
                    similarity = (img_features @ txt_features.T)

                print("clip: " + str(similarity.item()))
                if similarity >= 0.26:
                    draw_rect(image, mask["bbox"], class_name, (0,0,255))

                # draw mask
    #                 color = [randint(127, 255) for _ in range(3)]
    #                 alpha = 0.5
    #                 # draw mask overlay
    #                 colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
    #                 colored_mask = np.moveaxis(colored_mask, 0, -1)
    #                 masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    #                 image_overlay = masked.filled()
    #                 image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    #                 contours, _ = cv2.findContours(
    #                     np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    #                 )
    #                 cv2.drawContours(image, contours, -1, (255, 0, 0), 2)

        save_path = os.path.join(args.out_dir, file.split('/')[-1])
        cv2.imwrite(save_path, image)
        print("Saving to %s" % save_path)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    main(0, args)
