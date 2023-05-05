# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import cv2
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import pyvirtualcam

import torch
from torchvision import transforms

torch.backends.cudnn.benchmark = True

import networks

kernel_5 = np.ones((5, 5), np.uint8)
kernel_15 = np.ones((25, 25), np.uint8)

def noise_removal_and_expansion(img):
    blurred = cv2.medianBlur(img,3)
    org_frame_canny = cv2.dilate(blurred, kernel_5, iterations=3)
    org_frame_canny = cv2.morphologyEx(org_frame_canny, cv2.MORPH_CLOSE, kernel_15)
    org_frame_canny = cv2.dilate(org_frame_canny, kernel_15, iterations=1)
    org_frame_canny = cv2.morphologyEx(org_frame_canny, cv2.MORPH_CLOSE, kernel_15)
    org_frame_canny = cv2.dilate(org_frame_canny, kernel_15, iterations=2)
    org_frame_canny = cv2.morphologyEx(org_frame_canny, cv2.MORPH_CLOSE, kernel_15)
    final = cv2.dilate(org_frame_canny, kernel_15, iterations=1)

    return final


def test_simple():
    
    MODEL_NAME = 'mono+stereo_1024x320'

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = os.path.join("models", MODEL_NAME)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("-> Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("-> Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:

            vid = cv2.VideoCapture(0)
            
            while(True):
                ret, frame = vid.read()

                org_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                org_frame_blurred = cv2.medianBlur(org_frame,15)

                frame = org_frame.copy()
                
                input_image = frame
                input_image = pil.fromarray(np.uint8(frame)).convert('RGB')
                original_width, original_height = input_image.size
                input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                # PREDICTION
                input_image = input_image.to(device)
                features = encoder(input_image)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
                
                disp_resized_np = disp_resized.squeeze().cpu().numpy()
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                ORIGINAL_DEPTH_MAP = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

                ORIGINAL_DEPTH_MAP = cv2.medianBlur(ORIGINAL_DEPTH_MAP,3)

                gray_depth = cv2.cvtColor(ORIGINAL_DEPTH_MAP, cv2.COLOR_BGR2GRAY)
                gray_dilation = cv2.morphologyEx(gray_depth, cv2.MORPH_OPEN, kernel_5)

                org_frame_canny = cv2.Canny(org_frame,200,200)
                org_frame_canny =  noise_removal_and_expansion(org_frame_canny)
                
                depth_threshold_mask = gray_dilation >= 100
                org_frame_canny = cv2.normalize(org_frame_canny, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                frame[:,:,0] = frame[:,:,0] * org_frame_canny * depth_threshold_mask
                frame[:,:,1] = frame[:,:,1] * org_frame_canny * depth_threshold_mask
                frame[:,:,2] = frame[:,:,2] * org_frame_canny * depth_threshold_mask

                frame_combined = cv2.addWeighted(org_frame_blurred, 0.5, frame, 0.5, 0)
                
                blended_im = np.concatenate([ORIGINAL_DEPTH_MAP,frame_combined],axis = 1)
                color_blended = cv2.cvtColor(blended_im, cv2.COLOR_RGB2BGR)
                cv2.imshow('Depth Map + Frame', color_blended)

                # blended_canny = np.concatenate([colormapped_im_canny,org_frame_canny],axis = 1)
                # cv2.imshow('Canny', blended_canny)

                resized_output_frame = cv2.resize(frame_combined,(960,720))
                padded_output_frame = cv2.copyMakeBorder(resized_output_frame, 0, 0, 160, 160, cv2.BORDER_CONSTANT, None, value = 0)

                cam.send(padded_output_frame)
                cam.sleep_until_next_frame()


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            vid.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    test_simple()
