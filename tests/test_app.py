import numpy as np
import pytest

from ./app import count_pixels # relative import issue : add __init__.py

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
