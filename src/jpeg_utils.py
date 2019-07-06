import numpy as np


def split_88(image_data):
    blocks = []
    image_size = image_data.shape[0]
    for i in range(int(image_size / 8)):
        row = image_data[8 * i:8 * (i + 1), :]
        for j in range(int(image_size / 8)):
            col = row[:, 8 * j:8 * (j + 1)]
            blocks.append(col.astype(np.float32))
    return np.array(blocks)


def merge_88(blocks):
    block_count = blocks.shape[0]
    edge_len = int(np.sqrt(block_count))
    image_rows = []
    for i in range(edge_len):
        img_row = np.hstack(blocks[edge_len * i: edge_len * (i + 1), ...])
        image_rows.append(img_row)
    return np.vstack(image_rows)


def split_1616(image_data):
    blocks = []
    image_size = image_data.shape[0]
    for i in range(int(image_size / 16)):
        row = image_data[16 * i:16 * (i + 1), :]
        for j in range(int(image_size / 16)):
            col = row[:, 16 * j:16 * (j + 1)]
            blocks.append(col.astype(np.float32))
    return np.array(blocks)


def merge_1616(blocks):
    block_count = blocks.shape[0]
    edge_len = int(np.sqrt(block_count))
    img_size = 16 * edge_len
    image_rows = []
    for i in range(edge_len):
        img_row = np.hstack(blocks[edge_len * i: edge_len * (i + 1), ...])
        image_rows.append(img_row)
    return np.vstack(image_rows)



def zig_zag_flatten(a):
    return np.concatenate([np.diagonal(a[::-1, :], i)[::(2 * (i % 2) - 1)] for i in range(1 - a.shape[0], a.shape[0])])
