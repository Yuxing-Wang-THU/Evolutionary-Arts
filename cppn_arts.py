import torch
import torchvision
import numpy as np
from simple_cppn import CPPN, CPPN_config


# Calculate distances
def get_coordinates(dim_x, dim_y, scale=1.0, batch_size=1, o_point=(0, 0)):

    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = dim_x * dim_y
    x_range = scale * (np.arange(dim_x) - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5
    y_range = scale * (np.arange(dim_y) - (dim_y - 1) / 2.0) / (dim_y - 1) / 0.5
    center_point = (scale * (o_point[0] - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5,
                     scale * (o_point[1] - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5)

    x_mat = np.matmul(np.ones((dim_y, 1)), x_range.reshape((1, dim_x)))
    y_mat = np.matmul(y_range.reshape((dim_y, 1)), np.ones((1, dim_x)))
    r_mat = np.sqrt((x_mat-center_point[0]) * (x_mat-center_point[0]) + (y_mat-center_point[1]) * (y_mat-center_point[1]))
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()

torch.manual_seed(0)
np.random.seed(0)

# scale可以放大不同区域之间的差距
def main(pic_scale=1, o_point=(0.0,0.0)):
    config = CPPN_config(dim_z=64,dim_x=1000,dim_y=1000,dim_c=3,hidden_size=128, scale=pic_scale)
    
    print(config)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config.device = device
    print(f'device: {config.device}')

    model = CPPN(config).to(device)

    x, y, r = get_coordinates(config.dim_x, config.dim_y, config.scale,o_point=o_point)
    x, y, r = x.to(config.device), y.to(config.device), r.to(config.device)

    z = torch.randn(1, config.dim_z).to(device)
    scale = torch.ones((config.dim_x * config.dim_y, 1)).to(config.device)
    z_scaled = torch.matmul(scale, z)

    result = model(z_scaled, x, y, r)
    result = result.view(-1, config.dim_x, config.dim_y, config.dim_c).cpu()
    result = result.permute((0, 3, 1, 2))
    torchvision.utils.save_image(torchvision.utils.make_grid(result), f'art_{pic_scale}.jpg')


if __name__ == "__main__":
    main(100,o_point=(500,500))
