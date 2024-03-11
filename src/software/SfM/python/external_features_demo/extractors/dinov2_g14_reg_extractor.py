import torch
import torchvision
import torchvision.transforms as transforms

from extractors import BaseExtractor


class DINOv2G14RegExtractor(torch.nn.Module):
    '''
    The DINOv2 model with registers built upon ViT-G/14:
    https://github.com/facebookresearch/dinov2
    '''

    def __init__(self):
        super(DINOv2G14RegExtractor, self).__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitg14_reg'
        ).eval()
        # https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/data/transforms.py#L78
        self.preprocess = transforms.Compose([
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )
        ])

    def forward(self, img, keypoints):
        '''
        Extracts descriptors from the image.

        Parameters:
            img: torch.Tensor
                [1, 3, h, w] float32 tensor representing the input image. The values
                should have already been normalized to [0.0, 1.0].
            keypoints: torch.Tensor
                [n, 2] float32 tensor representing the keypoint coordinates. There
                are n keypoints, and the coordinates should be given in the order of
                (x, y).

        Return:
            descriptors: torch.Tensor
                [n, d] float32 tensor representing the extracted features for each
                keypoint.
        '''
        _, _, h, w = img.shape
        img = self.preprocess(img)
        with torch.no_grad():
            # Setting is_training=True allows us to obtain features with spatial
            # dimensions (before flattening).
            features = self.backbone(img, is_training=True)
            features = features['x_norm_patchtokens'].view(1, 16, 16, -1)
            features = features.permute(0, 3, 1, 2) # [1, 1536, 16, 16]
            grid = keypoints.view(1, keypoints.shape[0], 1, 2) # [1, n, 1, 2]
            grid = grid.to(features.device)
            # Normalize the grid to [-1, 1].
            grid[:, :, :, 0] = grid[:, :, :, 0] / w * 2.0 - 1.0
            grid[:, :, :, 1] = grid[:, :, :, 1] / h * 2.0 - 1.0
            descriptors = torch.nn.functional.grid_sample(
                features, grid, align_corners=True
            )
            descriptors = descriptors.squeeze(0).squeeze(-1)
            descriptors = descriptors.permute(1, 0) # [n, d]
        return descriptors

