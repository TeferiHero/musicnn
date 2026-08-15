import numpy as np

from musicnn.v2.dataset import create_fma_dataset, get_dataset, add_noise_to_fma_dataset, visualize_dataset, create_genre_test, draw_dataset_histogram


create_fma_dataset(True)
# add_noise_to_fma_dataset(0.1)

# get_dataset(train_noise_ratio=0.05)

# path = "fma/data/melspectrograms3/train/X_000000.npy"
# x = np.load(path)
# print("y parameters")
# print("shape:", x.shape)
# print("dtype:", x.dtype)
# print("ndim:", x.ndim)

# create_genre_test()
# draw_dataset_histogram()
# visualize_dataset()

