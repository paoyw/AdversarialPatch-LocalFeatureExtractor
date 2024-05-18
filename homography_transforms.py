import torch


def point_transform(H: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Homography transform on a list of points based on the transformation matrix.

    Args:
        H: The transformation matrix with shape [3, 3].
        points: A batch of points with shape [X, 2] or [X, 3].

    Returns:
        The points being transformed by the hommography matrix with shape[X, 3].
    """
    batch_size = points.shape[0]
    if points.shape[1] == 2:
        points = torch.hstack((points, torch.ones(batch_size, 1)))
    result = (H @ points.T).T
    return result[:, :2] / result[:, -1:]
