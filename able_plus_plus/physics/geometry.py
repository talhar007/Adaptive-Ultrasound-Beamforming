"""Array & imaging-grid geometry for the synthetic-aperture forward model.

All geometry (element positions, pixel grid, element-to-pixel distances) is
pre-computed once in ForwardModel.__init__ under torch.no_grad() and stored
as frozen buffers.  Nothing here is differentiable or trainable.
"""
import torch


def build_grids(x_lim, z_start, z_end, nx, nz, device):
    """Create a 2-D Cartesian imaging grid.

    Returns x_vec [nx], z_vec [nz], grid_X [nz, nx], grid_Z [nz, nx].
    """
    x_vec = torch.linspace(-x_lim, x_lim, nx, device=device)
    z_vec = torch.linspace(z_start, z_end, nz, device=device)
    grid_Z, grid_X = torch.meshgrid(z_vec, x_vec, indexing='ij')
    return x_vec, z_vec, grid_X, grid_Z


def linear_array(M, pitch, device):
    """x-coordinates of M evenly-spaced elements centred at x = 0.

    Returns [M, 1] tensor (column vector, for broadcasting against pixel grid).
    """
    return torch.linspace(-(M - 1) / 2, (M - 1) / 2, M, device=device).unsqueeze(1) * pitch


def element_pixel_distances(element_pos_x, grid_X, grid_Z):
    """Euclidean distance from every array element to every image-grid pixel.

    element_pos_x : [M, 1]        (element x-positions, z assumed = 0)
    grid_X, grid_Z: [nz, nx]      (pixel grid)
    returns        : [M, nx*nz]
    """
    px_x = grid_X.flatten()                                 # [P]
    px_z = grid_Z.flatten()                                 # [P]
    d_x  = px_x.unsqueeze(0) - element_pos_x               # [M, P]
    d_z  = px_z.unsqueeze(0).expand(element_pos_x.shape[0], -1)
    return torch.sqrt(d_x ** 2 + d_z ** 2)                 # [M, P]
