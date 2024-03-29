'''
kornia_deep_features.py
---------------------
This file extract the deep descriptors based on the SIFT feature points and match each keypoints based on two descriptors.
First use deep descriptors to find global position and use sift descriptors to find local position

Unlike kornia_demo.py, this file only supports feature extraction and does not
support matching.
'''
from pyvips import Image
from argparse import ArgumentParser
import kornia as K
import os
import torch
import torchvision.transforms as transforms
import threading
from tqdm import tqdm
import numpy as np
from itertools import combinations

import extractors
from kornia_demo import loadJSON, saveMatchesOpenMVG, loadFeatures
from sklearn.neighbors import KDTree


def saveDescriptorsOpenMVG(args, basename, descriptors):
    with open(os.path.join(args.matches, f'{basename}.desc'), 'wb') as desc:
        desc.write(len(descriptors).to_bytes(8, byteorder='little'))
        desc.write(descriptors.numpy().astype('<f4').tobytes())


def loadDescriptorsOpenMVG_int(args, basename):
    descriptors = []
    with open(os.path.join(args.matches, f'{basename}.desc'), 'rb') as desc:
        desc_content = desc.read()
        n = int.from_bytes(desc_content[:8], byteorder='little')
        descriptors = np.frombuffer(desc_content[8:], dtype="ubyte")
    descriptors = np.reshape(descriptors, (n, -1))
    return descriptors


def loadDescriptorsOpenMVG_float(args, basename):
    descriptors = []
    with open(os.path.join(args.matches, f'{basename}.desc'), 'rb') as desc:
        desc_content = desc.read()
        n = int.from_bytes(desc_content[:8], byteorder='little')
        descriptors = np.frombuffer(desc_content[8:], dtype="<f4")
    descriptors = np.reshape(descriptors, (n, -1))
    return descriptors


def loadFeaturesOpenMVG(args, basename):
    keypoints = []
    with open(os.path.join(args.matches, f'{basename}.feat'), 'r') as feat:
        for l in feat:
            l = l.strip().split()
            keypoints.append([float(l[0]), float(l[1])])
    keypoints = np.array(keypoints)
    return keypoints


def featureExtraction(args):
    '''
    This function is mostly identical to featureExtraction() in kornia_demo.py.
    The main difference is that the DISK feature descriptoirs are replaced by deep
    features. The keypoint coordinates and the detection scores are unchanged.
    '''
    print('Extracting deep features...')
    if(not os.path.exists(os.path.join(args.matches, "deep_desc"))):
        os.mkdir(os.path.join(args.matches, "deep_desc"))

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

        ################################################
        # Changes to kornia_demo.py starts here        #
        ################################################
        keypoints = loadFeaturesOpenMVG(args, basename)
        keypoints = torch.tensor(keypoints.astype(np.float32)).to(device)

        descriptors = deep_extractor(img, keypoints * scale).to('cpu')
        ################################################
        # Changes to kornia_demo.py ends here          #
        ################################################

        threading.Thread(target=lambda: saveDescriptorsOpenMVG(
            args, os.path.join("deep_desc", basename), descriptors
        )).start()


def featureMatching(args, device, view_ids):
    print('Matching DISK features with two step...')
    putative_matches = []
    lastidx = -1
    for image1_index, image2_index in tqdm((np.loadtxt(args.pair_list, dtype=np.int32) if args.pair_list != None else np.asarray([*combinations(view_ids, 2)], dtype=np.int32))):
        keyp1 = loadFeaturesOpenMVG(args, os.path.splitext(view_ids[image1_index])[0])
        keyp2 = loadFeaturesOpenMVG(args, os.path.splitext(view_ids[image2_index])[0])
        desc1_sift = loadDescriptorsOpenMVG_int(args, os.path.splitext(view_ids[image1_index])[0]).astype(int)
        desc2_sift = loadDescriptorsOpenMVG_int(args, os.path.splitext(view_ids[image2_index])[0]).astype(int)
        desc1_deep = loadDescriptorsOpenMVG_float(args, os.path.join("deep_desc", os.path.splitext(view_ids[image1_index])[0]))
        desc2_deep = loadDescriptorsOpenMVG_float(args, os.path.join("deep_desc", os.path.splitext(view_ids[image2_index])[0]))
        idxs = []
        if(lastidx != image1_index):
            tree = KDTree(desc1_deep)
            tree_sift = KDTree(keyp1)
            lastidx = image1_index
        _, idx_neigh = tree_sift.query(keyp1, k = 50)
        _, idx = tree.query(desc2_deep, k=1)
        idx = idx[:, 0]
        for j in range(len(idx)):
            idxi = idx_neigh[idx[j]]
            idxi = np.unique(idxi)
            disij = np.linalg.norm(desc1_sift[idxi] - desc2_sift[j], axis=1)
            #min_idx = idxi[np.argmin(np.linalg.norm(desc1_sift[idxi] - desc2_sift[j], axis=1))]
            i1, i2 = np.argpartition(disij, 1)[:2]
            if(disij[i1] < disij[i2]*0.8):
                i1 = idxi[i1]
                idxs.append([i1, j])

        putative_matches.append([image1_index, image2_index, np.array(idxs).astype(np.int32)])

    print('Saving putative matches...')
    saveMatchesOpenMVG(args, putative_matches)


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
    parser.add_argument('--preset', choices=['BOTH','EXTRACT','MATCH'], default='BOTH', help='Preset to run')
    parser.add_argument('-p', '--pair_list', type=str, help='Path to the pair file')

    args = parser.parse_args()

    view_ids, image_paths = loadJSON(args)
    if args.output == None:
        args.output = os.path.join(args.matches, 'matches.putative.bin')

    if args.force_cpu:
        device = torch.device('cpu')
    else:
        device = K.utils.get_cuda_device_if_available()
        device = "cuda:1"

    #disk = K.feature.DISK().from_pretrained('depth').to(device)
    #print('Loaded DISK model')

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
        if args.preset == 'EXTRACT' or args.preset == 'BOTH':
            featureExtraction(args)
        if args.preset == 'MATCH' or args.preset == 'BOTH':
            featureMatching(args, device, view_ids)
