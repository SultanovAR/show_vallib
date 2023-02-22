import matplotlib.pyplot as plt
from PIL import Image


def plt2PIL(fig):
    fig.canvas.draw()
    pil_image = Image.frombytes(
        'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return pil_image
