import torch
import torchvision
import torchvision.transforms as transforms

from extractors import BaseExtractor


class DeepLabv3Extractor(BaseExtractor):
    '''
    The DeepLab ResNet101 model:
    https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html
    '''

    def __init__(self):
        super(DeepLabv3Extractor, self).__init__()
        weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        self.backbone = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=weights
        ).backbone.eval()
        # https://github.com/facebookresearch/dinov2/blob/c3c2683a13cde94d4d99f523cf4170384b00c34c/dinov2/data/transforms.py#L78
        self.preprocess = transforms.Compose([
            transforms.Resize(
                (520), interpolation=transforms.InterpolationMode.BILINEAR
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
            features = self.backbone(img)['out'] # [1, 2048, h', w']
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

