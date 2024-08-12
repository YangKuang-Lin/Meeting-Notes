import torch

def forward_process(x0, t):
    """
    The forward process to noise the image.    

    Parameters
    ----------
    x0: tensor
        Original image.
    t: tensor
        Timestep.

    Returns
    -------
    x: tensor
        Noised image.

    Notes
    -----
    .. [1] Source Code, https://qiita.com/CabbageRoll/items/7c79ae63ba417271226e

    """

    if not t:
        t = torch.randint(low=1, high=t, size=(x0.shape[0],))

    # Generate noise.
    noise = torch.randn_like(x0)

    # Generate beta with linear interpolation from 1e-4 to 0.02.
    b = torch.linspace(1e-4, 0.02, t)

    # Compute alpha.
    a = 1 - b
    a = torch.cumprod(a, dim=0).reshape(-1, 1, 1, 1)

    # Compute noised image.
    x = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise

    return x
