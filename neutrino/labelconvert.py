
_labels = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

label_idx = {}
idx_lable = {}

idx = 0
for l in _labels:
    label_idx[l] = idx
    idx_lable[idx] = l
    idx += 1


def get_label_idx(label):
    return label_idx[label]


def get_idx_label(idx):
    return idx_lable[idx]
