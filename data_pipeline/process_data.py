import numpy as np
from tqdm.auto import tqdm
import h5py
from pathlib import Path


def merge_files_from_datagen(data_path: str, output_file: str):
    generated_files = list(Path(data_path).rglob("*.hdf5"))
    # Calculate total amount
    N = 0
    max_cuboids = 0
    max_cylinders = 0
    for fn in tqdm(generated_files):
        with h5py.File(str(fn)) as g:
            N += len(g["global_solutions"])
            if "cuboid_centers" in g.keys():
                max_cuboids = max(max_cuboids, g["cuboid_centers"].shape[1])
            if "cylinder_centers" in g.keys():
                max_cylinders = max(max_cylinders, g["cylinder_centers"].shape[1])
    print(f"Found {N} global solutions")
    # with h5py.File(str(scene_path / 'global_data.hdf5'), 'w') as f:
    with h5py.File(output_file, "w-") as f:
        with h5py.File(str(generated_files[0])) as g:
            for k in g.keys():
                if "cuboid" in k:
                    f.create_dataset(k, (N, max_cuboids, g[k].shape[2]))
                elif "cylinder" in k:
                    f.create_dataset(k, (N, max_cylinders, g[k].shape[2]))
                else:
                    f.create_dataset(k, (N, *g[k].shape[1:]))
        end = 0
        for fn in tqdm(generated_files):
            with h5py.File(str(fn)) as g:
                for k in g.keys():
                    f[k][end : end + len(g[k]), : g[k].shape[1], ...] = g[k][...]
                end += len(g[k])


def extract_expert_data(input_file: str, output_file: str):
    with h5py.File(input_file) as g:
        expert_indices = np.nonzero(
            ~np.all(g["expert_solutions"] == np.zeros((50, 7)), axis=(1, 2))
        )[0]
        N = len(expert_indices)
        print(f"Found {N} expert demonstrations")
        with h5py.File(output_file, "w-") as f:
            for k in g.keys():
                f.create_dataset(k, (N, *g[k].shape[1:]))
            for ii, jj in enumerate(tqdm(expert_indices)):
                for k in g.keys():
                    f[k][ii, ...] = g[k][jj, ...]


def downsize_and_split_global(path, train_size, val_size, test_size):
    """
    For all datasets, expert data will be removed (and a simple 1 or 0 will mark if there was an expert)

    Train set is sampled from everything
    Validation set is sampled from everything
    No Experts Test set is sampled only from problems that cannot be solved by expert
    All Experts Test set is sampled from problems that can be solved by expert
    """
    with h5py.File(str(path / "global_data.hdf5")) as f:
        assert train_size + val_size + 2 * test_size < len(f["cuboid_centers"])
        print(f["expert_solutions"].shape)
        without_expert = np.all(f["expert_solutions"] == np.zeros((50, 7)), axis=(1, 2))

        non_expert_indices = np.nonzero(without_expert)[0]
        expert_indices = np.nonzero(~without_expert)[0]

        no_experts_test_indices = np.random.choice(
            non_expert_indices, size=test_size, replace=False
        )
        all_experts_test_indices = np.random.choice(
            expert_indices, size=test_size, replace=False
        )
        index_mask = np.ones(len(f["cuboid_centers"]), dtype=bool)
        index_mask[all_experts_test_indices] = False
        index_mask[no_experts_test_indices] = False

        indices = np.random.choice(
            np.arange(len(f["cuboid_centers"]))[index_mask],
            size=train_size + val_size,
            replace=False,
        )
        train_indices, val_indices = (
            indices[:train_size],
            indices[train_size:],
        )
        assert (
            len(train_indices)
            + len(val_indices)
            + len(all_experts_test_indices)
            + len(no_experts_test_indices)
            == train_size + val_size + 2 * test_size
        )
        (
            train_indices,
            val_indices,
            all_experts_test_indices,
            no_experts_test_indices,
        ) = (
            np.sort(train_indices),
            np.sort(val_indices),
            np.sort(all_experts_test_indices),
            np.sort(no_experts_test_indices),
        )
        if val_size > 0:
            with h5py.File(str(path / "global" / "val" / "val.hdf5"), "w") as g:
                for k in f.keys():
                    if k == "expert_solutions":
                        g.create_dataset("has_expert", (val_size, 1))
                    elif k == "global_solutions":
                        g.create_dataset(
                            "robot_configurations", (val_size, *f[k].shape[1:])
                        )
                    else:
                        g.create_dataset(k, (val_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(val_indices)):
                    for k in g.keys():
                        if k == "has_expert":
                            g[k][ii] = int(np.any(f["expert_solutions"][jj]))
                        else:
                            if k == "robot_configurations":
                                ds = f["global_solutions"]
                            else:
                                ds = f[k]
                            g[k][ii, ...] = ds[jj, ...]
        if test_size > 0:
            with h5py.File(
                str(path / "global" / "test" / "all_experts_test.hdf5"), "w"
            ) as g:
                for k in f.keys():
                    if k == "expert_solutions":
                        g.create_dataset("has_expert", (test_size, 1))
                    elif k == "global_solutions":
                        g.create_dataset(
                            "robot_configurations", (test_size, *f[k].shape[1:])
                        )
                    else:
                        g.create_dataset(k, (test_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(all_experts_test_indices)):
                    for k in g.keys():
                        if k == "has_expert":
                            g[k][ii] = int(np.any(f["expert_solutions"][jj]))
                        else:
                            if k == "robot_configurations":
                                ds = f["global_solutions"]
                            else:
                                ds = f[k]
                            g[k][ii, ...] = ds[jj, ...]
            with h5py.File(
                str(path / "global" / "test" / "no_experts_test.hdf5"), "w"
            ) as g:
                for k in f.keys():
                    if k == "expert_solutions":
                        g.create_dataset("has_expert", (test_size, 1))
                    elif k == "global_solutions":
                        g.create_dataset(
                            "robot_configurations", (test_size, *f[k].shape[1:])
                        )
                    else:
                        g.create_dataset(k, (test_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(no_experts_test_indices)):
                    for k in g.keys():
                        if k == "has_expert":
                            g[k][ii] = int(np.any(f["expert_solutions"][jj]))
                        else:
                            if k == "robot_configurations":
                                ds = f["global_solutions"]
                            else:
                                ds = f[k]
                            g[k][ii, ...] = ds[jj, ...]
        if train_size > 0:
            with h5py.File(str(path / "global" / "train" / "train.hdf5"), "w") as g:
                for k in f.keys():
                    if k == "expert_solutions":
                        g.create_dataset("has_expert", (train_size, 1))
                    elif k == "global_solutions":
                        g.create_dataset(
                            "robot_configurations", (train_size, *f[k].shape[1:])
                        )
                    else:
                        g.create_dataset(k, (train_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(train_indices)):
                    for k in g.keys():
                        if k == "has_expert":
                            g[k][ii] = int(np.any(f["expert_solutions"][jj]))
                        else:
                            if k == "robot_configurations":
                                ds = f["global_solutions"]
                            else:
                                ds = f[k]
                            g[k][ii, ...] = ds[jj, ...]


def merge_files(files, output_file):
    if len(files) == 0:
        return
    max_cylinders = 0
    max_cuboids = 0
    N = 0
    for fn in files:
        with h5py.File(str(fn)) as f:
            N += len(f["robot_configurations"])
            if "cylinder_centers" in f.keys():
                max_cylinders = max(f["cylinder_centers"].shape[1], max_cylinders)
            if "cuboid_centers" in f.keys():
                max_cuboids = max(f["cuboid_centers"].shape[1], max_cylinders)
    with h5py.File(str(output_file), "w") as g:
        with h5py.File(files[0]) as f:
            for k in f.keys():
                if "cuboid" in k:
                    g.create_dataset(k, (N, max_cuboids, *f[k].shape[2:]))
                elif "cylinder" in k:
                    g.create_dataset(k, (N, max_cylinders, *f[k].shape[2:]))
                else:
                    g.create_dataset(k, (N, *f[k].shape[1:]))

        idx = 0
        for fn in tqdm(files):
            with h5py.File(str(fn)) as f:
                n = len(f["robot_configurations"])
                for ii in trange(0, n, 10000):
                    if ii < (n // 10000) * 10000:
                        for k in f.keys():
                            if "cuboid" in k:
                                g[k][
                                    idx + ii : idx + ii + 10000,
                                    : f["cuboid_centers"].shape[1],
                                    ...,
                                ] = f[k][ii : ii + 10000, ...]
                            elif "cylinder" in k:
                                g[k][
                                    idx + ii : idx + ii + 10000,
                                    : f["cylinder_centers"].shape[1],
                                    ...,
                                ] = f[k][ii : ii + 10000, ...]
                            else:
                                g[k][idx + ii : idx + ii + 10000, ...] = f[k][
                                    ii : ii + 10000, ...
                                ]
                    else:
                        for k in f.keys():
                            if "cuboid" in k:
                                g[k][
                                    idx + ii : idx + n,
                                    : f["cuboid_centers"].shape[1],
                                    ...,
                                ] = f[k][ii:, ...]
                            elif "cylinder" in k:
                                g[k][
                                    idx + ii : idx + n,
                                    : f["cylinder_centers"].shape[1],
                                    ...,
                                ] = f[k][ii:, ...]
                            else:
                                g[k][idx + ii : idx + n, ...] = f[k][ii:, ...]
                idx += n


def merge_expert_data():
    merge_files(
        list(Path("/workspace/data/mpinets/").rglob("*expert/train/train.hdf5")),
        "/workspace/data/mpinets/expert_dataset/train/train.hdf5",
    )
    merge_files(
        list(Path("/workspace/data/mpinets").rglob("*expert/val/val.hdf5")),
        "/workspace/data/mpinets/expert_dataset/val/val.hdf5",
    )
    merge_files(
        list(Path("/workspace/data/mpinets").rglob("*expert/test/test.hdf5")),
        "/workspace/data/mpinets/expert_dataset/test/test.hdf5",
    )
