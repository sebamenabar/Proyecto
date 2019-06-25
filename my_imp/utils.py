from torch.utils.data import Subset


curriculum_strategy = [
    (0, 3, 4),
    (5, 3, 6),
    (10, 3, 8),
    (15, 4, 8),
    (25, 4, 12),
    (35, 5, 12),
    (45, 6, 12),
    (55, 7, 16),
    (65, 8, 20),
    (75, 9, 22),
    (90, 10, 25),
    (1e9, None, None)
]


def build_curriculum(dataset, epoch):
    subset_idxs = []
    for ci, (min_epoch, max_scene_size, max_program_size) in enumerate(curriculum_strategy):
        if min_epoch < epoch <= curriculum_strategy[ci + 1][0]:
            for j in range(len(dataset)):
                data_info = dataset._get_metainfo(j)
                if (data_info['scene']['scene_size'] <= max_scene_size) and (data_info['question']['program_size'] <= max_program_size):
                    subset_idxs.append(j)
            break

    return Subset(dataset, subset_idxs)
