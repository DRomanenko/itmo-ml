# Kernels
def kernel_uniform(u):
    return 0.5 if abs(u) <= 1 else 0.0


def kernel_triangular(u):
    return max(0, 1 - abs(u))


def kernel_epanechnikov(u):
    return max(0, 0.75 * (1 - u * u))


def kernel_quartic(u):
    return max(0, 15 / 16 * ((1 - u * u) ** 2))


kernels = {
    "uniform": kernel_uniform,
    "triangular": kernel_triangular,
    "epanechnikov": kernel_epanechnikov,
    "quartic": kernel_quartic,
}