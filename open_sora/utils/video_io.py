# Copyright © 2024 Apple Inc.

import av
import mlx.core as mx
import numpy as np


def save_video(x, save_path=None, fps=8):
    """
    Save an MLX array as a video.

    Args:
        x (mx.array): shape [T, H, W, C]
    """

    def normalize(x):
        x = (mx.clip(x, a_min=-1.0, a_max=1.0) + 1.0) * (255.0 / 2.0)
        x = mx.clip(x + 0.5, a_min=0, a_max=255).astype(mx.uint8)
        return x

    x = np.array(normalize(x))

    with av.open(save_path, mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = x.shape[2]
        stream.height = x.shape[1]
        stream.pix_fmt = "yuv420p"

        for img in x:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
