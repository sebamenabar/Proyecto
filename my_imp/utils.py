from torch.utils.data import Subset


curriculum_strategy = [
    (0, 3, 4),
    (10, 3, 6),
    (20, 3, 8),
    (30, 4, 8),
    (40, 4, 12),
    (50, 5, 12),
    (60, 6, 12),
    (70, 7, 16),
    (80, 8, 20),
    (90, 9, 22),
    (100, 10, 25),
    (1e9, None, None)
]


def build_curriculum(dataset, epoch):
    subset_idxs = []
    for ci, (min_epoch, max_scene_size, max_program_size) in enumerate(curriculum_strategy):
        if min_epoch < epoch <= curriculum_strategy[ci + 1][0]:
            print(f'Curriculum with {max_scene_size} max scene size and {max_program_size} max program size')
            for j in range(len(dataset)):
                data_info = dataset._get_metainfo(j)
                if (data_info['scene']['scene_size'] <= max_scene_size) and (data_info['question']['program_size'] <= max_program_size):
                    subset_idxs.append(j)
            break

    return Subset(dataset, subset_idxs)
