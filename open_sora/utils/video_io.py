import cv2
import mlx.core as mx
import numpy as np

def save_sample(x, save_path=None, fps=8):
    """
    Args:
        x (mx.array): shape [T, H, W, C]
    """
    save_path += ".mp4"

    def normalize(x):
        x = (mx.clip(x, a_min=-1.0, a_max=1.0) + 1.0) * (255.0 / 2.0)
        x = mx.clip(x + 0.5, a_min=0, a_max=255).astype(mx.uint8)
        return x

    x = np.array(normalize(x))
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (x.shape[2], x.shape[1]),
    )
    for frame in x:
        out.write(frame)
    out.release()
