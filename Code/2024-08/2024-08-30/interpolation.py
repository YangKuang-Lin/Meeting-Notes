def interpotation(x1, x2, l, t):
    """
    The function returns the interpolated image.

    Parameters
    ----------
    x1: tensor
        Image x0.
    x2: tensor
        Image x'0
    l: float
        Lambda.
    t: float
        Time.

    Returns
    -------
    x: tensor
        The interpolated image.

    Notes
    -----
    .. [1] Source Code, https://nn.labml.ai/diffusion/ddpm/evaluate.html

    """

    # diffusion.q denotes the non-parametric forward process to
    # add noise into the image for timestep t.
    x = (1 - l) * diffusion.q(x1, t) + l * diffusion.q(x2, t)

    return x
