'''
kornia_deep_features.py
---------------------
This file replaces the DISK descriptors in kornia_demo.py by deep descriptors.
It starts by using DISK to obtain the keypoint coordinates and then extracts the
deep features at the DISK coordinates.

Unlike kornia_demo.py, this file only supports feature extraction and does not
support matching.
'''

from argparse import ArgumentParser
import kornia as K
import os
from pyvips import Image
import torch
import torchvision.transforms as transforms
import threading
from tqdm import tqdm

import extractors
from kornia_demo import loadJSON, saveFeaturesOpenMVG


def saveDescriptorsOpenMVG(args, basename, descriptors):
    with open(os.path.join(args.matches, f'{basename}.desc'), 'wb') as desc:
        desc.write(len(descriptors).to_bytes(8, byteorder='little'))
        desc.write(descriptors.numpy().astype('<f4').tobytes())


def featureExtraction(args):
    '''
    This function is mostly identical to featureExtraction() in kornia_demo.py.
    The main difference is that the DISK feature descriptoirs are replaced by deep
    features. The keypoint coordinates and the detection scores are unchanged.
    '''
    print('Extracting deep features...')
    for image_path in tqdm(image_paths):
        img = Image.new_from_file(image_path, access='sequential')
        basename = os.path.splitext(os.path.basename(image_path))[0]

        if img.width % 2 != 0 or img.height % 2 != 0:
            img = img.crop(
                0,
                0,
                img.width if img.width % 2 == 0 else img.width - 1,
                img.height if img.height % 2 == 0 else img.height - 1
            )

        max_res = args.max_resolution
        img_max, img_ratio = max(img.width, img.height), img.width / img.height

        ratio = 0
        while not ratio:
            scale = max_res / img_max
            scaled_width = round(img.width * scale)
            scaled_height = round(img.height * scale)
            if img_ratio == scaled_width / scaled_height:
                ratio = 1
            else:
                max_res -= 1

        img = transforms.ToTensor()(
            img.resize(scale, kernel='linear').numpy()
        )[None, ...].to(device)

        features = disk(
            img,
            n=args.max_features,
            window_size=args.window_size,
            score_threshold=args.score_threshold,
            pad_if_not_divisible=True)[0].to('cpu')

        ################################################
        # Changes to kornia_demo.py starts here        #
        ################################################
        features.descriptors = deep_extractor(img, features.keypoints).to('cpu')
        keypoints = torch.div(features.keypoints, scale)
        ################################################
        # Changes to kornia_demo.py ends here          #
        ################################################

        threading.Thread(target=lambda: saveFeaturesOpenMVG(
            args, basename, keypoints
        )).start()
        threading.Thread(target=lambda: saveDescriptorsOpenMVG(
            args, basename, features.descriptors
        )).start()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, required=True, help='Path to the sfm_data file'
    )
    parser.add_argument(
        '--max_resolution', type=int, default=1024, help='Max image resolution'
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=4096,
        help='Max number of features to extract'
    )
    parser.add_argument(
        '-m',
        '--matches',
        type=str,
        required=True,
        help='Path to the matches directory'
    )
    parser.add_argument(
        '-o', '--output', type=str, help='Path to the output matches file'
    )
    parser.add_argument(
        '--force_cpu', action='store_true', help='Force device to CPU'
    )
    # DISK
    parser.add_argument(
        '--window_size',
        type=int,
        default=5,
        help='DISK Non Maximum Suppression (NMS) radius (Must be odd)'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.01,
        help='DISK keypoint detector confidence threshold'
    )
    # Deep extractor
    parser.add_argument(
        '--deep_descriptor_type',
        type=str,
        default='DeepLabv3',
        help='Deep extractor type, can be either DeepLabv3 or DINOv2'
    )
    args = parser.parse_args()

    view_ids, image_paths = loadJSON(args)
    if args.output == None:
        args.output = os.path.join(args.matches, 'matches.putative.bin')

    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = K.utils.get_cuda_device_if_available()

    disk = K.feature.DISK().from_pretrained('depth').to(device)
    print('Loaded DISK model')

    ################################################
    # Changes to kornia_demo.py starts here        #
    ################################################
    if args.deep_descriptor_type == 'DeepLabv3':
        deep_extractor = extractors.DeepLabv3Extractor().to(device)
        print('Loaded DeepLabv3 model')
    elif args.deep_descriptor_type == 'DINOv2':
        deep_extractor = extractors.DINOv2G14RegExtractor().to(device)
        print('Loaded DINOv2 model')
    else:
        raise ValueError('The deep descriptor type is not supported')
    ################################################
    # Changes to kornia_demo.py ends here          #
    ################################################

    with torch.inference_mode():
        featureExtraction(args)
