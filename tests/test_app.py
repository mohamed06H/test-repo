import numpy as np
import pytest

#from ..app import count_pixels # relative import issue : add __init__.py

def count_pixels(image, pixel:tuple[int, int, int]):
    """ Counts the occurences of an RGB value in an image
    """
    pixel_count = np.sum(np.all(image==pixel, axis=-1)) # Condition must match all the three RGB layers -> axis = 2 or -1
    return pixel_count

def test_count_pixels():
    # Test case 1: Single color image
    image = np.array([
        [[255, 255, 255], [255, 255, 255]],
        [[255, 255, 255], [255, 255, 255]]
    ])
    pixel = (255, 255, 255)
    assert count_pixels(image, pixel) == 4

if __name__ == "__main__":
    pytest.main()
