import torch
import torch.nn.functional as F


def affine_transform(inp, matrix, mode='bilinear'):
    matrix = torch.unsqueeze(matrix, dim=0).expand(inp.size(0), 2, 3)
    grid = F.affine_grid(matrix,
                         inp.size(),
                         align_corners=False)
    trans = F.grid_sample(inp.type(torch.float),
                          grid.type(torch.float),
                          align_corners=False,
                          mode=mode)
    return trans
