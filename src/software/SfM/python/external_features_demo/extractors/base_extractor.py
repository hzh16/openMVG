import torch

class BaseExtractor(torch.nn.Module):
    def __init__(self):
        super(BaseExtractor, self).__init__()

    def extract(self, img, keypoints):
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
        raise NotImplementedError()
