"""
Toy example of the SVD convolution

This module provides a toy example of the SVD (Singular Value Decomposition) convolution.
It includes two classes: ClassicConvolution and SVDConvolution,
as well as an ImageGenerator class for generating random images for testing.

Classes:
- ClassicConvolution: Represents a classic convolution.
- FFTConvolution: Represents a FFT convolution.
- SVDConvolution: Represents the SVD approximate convolution.
- ImageGenerator: Generates random images for testing.

"""

import random
import scipy as sp
import numpy as np


class ClassicConvolution:
    """
    The class that represents a classic convolution
    """

    def __init__(self, arr: np.array, kernel: np.array) -> None:
        """
        Initialize the ClassicConvolution class.

        Parameters:
        - arr (np.array): The input array.
        - kernel (np.array): The convolution kernel.
        """
        self.arr = np.copy(arr)
        self.kernel = np.copy(kernel)

    @property
    def convolution(self) -> np.array:
        """
        Perform the convolution operation.

        Returns:
        - np.array: The result of the convolution.
        """
        return sp.signal.convolve2d(self.arr, self.kernel)

class FFTConvolution:
    """
    The class that represents a FFT convolution
    """

    def __init__(self, arr: np.array, kernel: np.array) -> None:
        """
        Initialize the ClassicConvolution class.

        Parameters:
        - arr (np.array): The input array.
        - kernel (np.array): The convolution kernel.
        """
        self.arr = np.copy(arr)
        self.kernel = np.copy(kernel)

    @property
    def convolution(self) -> np.array:
        """
        Perform the convolution operation with FFT.

        Returns:
        - np.array: The result of the convolution.
        """
        return sp.signal.fftconvolve(self.arr, self.kernel)

class SVDConvolution:
    """
    The class that represents the SVD approximate convolution
    """

    def __init__(
        self, arr: np.array, kernel: np.array, rows: int | None = None
    ) -> None:
        """
        Initialize the SVDConvolution class.

        Parameters:
        - arr (np.array): The input array.
        - kernel (np.array): The convolution kernel.
        """
        self.rank = rows
        self.arr = np.copy(arr)
        self._kernel = np.copy(kernel)
        self._u, self._s, self._v = np.linalg.svd(self.kernel)
        self.rank = self.rank or np.linalg.matrix_rank(self.kernel)

    @property
    def kernel(self) -> np.array:
        """
        Get the kernel matrix.

        Returns:
        - np.array: The kernel matrix.
        """
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel: np.array):
        """
        Set the kernel matrix and perform Singular Value Decomposition (SVD).

        Parameters:
        - new_kernel (np.array): The new kernel matrix.

        Returns:
        - None
        """
        self._kernel = np.copy(new_kernel)
        self._u, self._s, self._v = np.linalg.svd(self.kernel)
        self.rank = self.rank or np.linalg.matrix_rank(self.kernel)

    @property
    def convolution(self) -> np.array:
        """
        Perform the convolution operation.

        Parameters:
        - rows (int | None): The number of rows to consider in the SVD
        approximation. If None, half of the rows of the SVD matrix will be used.

        Returns:
        - np.array: The result of the approximate convolution.
        """

        u, s, v = self._u, self._s, self._v

        hterms = np.array(
            [
                [
                    s[j] * np.convolve(u[:, j], self.arr[i])
                    for i in range(self.arr.shape[0])
                ]
                # for j in range(u.shape[0])
                for j in range(self.rank)
            ]
        )

        terms = np.array(
            [
                np.array(
                    [np.convolve(v[j], hterm.T[i]) for i in range(hterm.T.shape[0])]
                ).T
                for j, hterm in enumerate(hterms)
            ]
        )

        return terms.sum(axis=0)


class ImageGenerator:
    """
    Generator of random images for testing
    """

    def __init__(
        self, size: tuple[int, int] = (16, 16), population_factor: int = 50
    ) -> None:
        """
        Initialize the ImageGenerator class.

        Parameters:
        - size (tuple[int, int]): The size of the generated image.
        - population_factor (int): The population factor for generating random values.
        """
        self.size = size
        self.population_factor = population_factor
        self.generate_image()

    def generate_image(self, bounds: tuple[int, int] = (-100, 100)):
        """
        Generate a random image.

        Parameters:
        - bounds (tuple[int, int]): The bounds for the random values.

        Returns:
        - None
        """
        self.image = np.zeros(self.size).reshape(self.size)
        for x in range(
            0, int(self.size[0] * self.size[1] * self.population_factor / 100)
        ):
            found = False
            while not found:
                coord_x = random.randint(0, self.size[0] - 1)
                coord_y = random.randint(0, self.size[1] - 1)
                if self.image[coord_x, coord_y] != 0:
                    continue
                found = True
                self.image[coord_x, coord_y] = random.uniform(bounds[0], bounds[1] + 1)


if __name__ == "__main__":
    import timeit

    r = range(-2, 3)
    x, y = np.meshgrid(r, r)
    kernel = x ** 2 + y ** 2 < 4

    img_gen = ImageGenerator((500, 500))

    conv = ClassicConvolution(img_gen.image, kernel)
    res = conv.convolution
    t_classic = timeit.timeit(lambda: conv.convolution, number=1000)
    print(f"Classic: {t_classic}")

    fft_conv = FFTConvolution(img_gen.image, kernel)
    res = fft_conv.convolution
    t_classic = timeit.timeit(lambda: conv.convolution, number=1000)
    print(f"FFT conv: {t_classic}")

    conv_approx = SVDConvolution(img_gen.image, kernel)
    res_svd = conv_approx.convolution
    t_svd = timeit.timeit(lambda: conv_approx.convolution, number=1000)
    print(f"SVD: {t_svd}")

    print(abs(res_svd - res).sum())
