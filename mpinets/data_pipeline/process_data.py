# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
from tqdm.auto import tqdm, trange
import h5py
from pathlib import Path
from typing import List
from argparse import ArgumentParser, RawTextHelpFormatter
import sys


def merge_files(files: List[Path], output_file: str):
    """
    Will merge a bunch of hdf5 dataset files into a single file.

    :param data_path List[Path]: A list of files to merge
    :param output_file str: The file to output. Should be something like `oogyboogy.hdf5`
    """
    # Calculate total amount
    N = 0
    max_cuboids = 0
    max_cylinders = 0
    for fn in tqdm(files):
        with h5py.File(str(fn)) as f:
            N += len(f["global_solutions"])
            if "cuboid_centers" in f.keys():
                max_cuboids = max(max_cuboids, f["cuboid_centers"].shape[1])
            if "cylinder_centers" in f.keys():
                max_cylinders = max(max_cylinders, f["cylinder_centers"].shape[1])
    with h5py.File(output_file, "w-") as g:
        with h5py.File(str(files[0])) as f:
            for k in f.keys():
                if "cuboid" in k:
                    g.create_dataset(k, (N, max_cuboids, f[k].shape[2]))
                elif "cylinder" in k:
                    g.create_dataset(k, (N, max_cylinders, f[k].shape[2]))
                else:
                    g.create_dataset(k, (N, *f[k].shape[1:]))
        idx = 0
        for fn in tqdm(files):
            with h5py.File(str(fn)) as f:
                n = len(f["cuboid_centers"])
                # Do the copying in 10000 data point chunks for memory reasons
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


def merge_data_pipeline_files(data_path: str, output_file: str):
    """
    Will merge a bunch of final output files from the `gen_data.py` script. These files
    should all correspond to the same problem paradigm (i.e. with or without neutral poses)
    of the same scene type (i.e. dresser, cubby, merged cubby, or tabletop). For example,
    they should all be dresser scenes going to/from a neutral pose.

    :param data_path str: A directory containing all the merged files (the final outputs)
                          of the `gen_data.py` script. Note that the script's final output
                          is always named `all_data.hdf5`, so these files should either be
                          renamed or kept in unique folders that live within `data_path`.
                          This function will do a recursive search within this folder to find
                          all `*.hdf5` files.
    :param output_file str: The file to output. Should be something like `blahblahblah.hdf5`
    """
    generated_files = list(Path(data_path).rglob("*.hdf5"))
    merge_files(generated_files, output_file)


def extract_hybrid_expert_data(input_file: str, output_file: str):
    """
    Takes a file generated by `merge_data_pipeline_files` and removes all trajectories
    that do not have a hybrid expert solution. This is necessary if you want to train on
    only hybrid solutions because it's much faster to do this once during processing than
    to rely on the dataloader to do it.

    :param input_file str: The output of `merge_data_pipeline_files`. Should be something like
                           `blahblahblah.hdf5`
    :param output_file str: The file to output with only the problems with hybrid expert
                            demonstrations. Should be something like `somethingsomething.hdf5`
    """
    with h5py.File(input_file) as g:
        hybrid_indices = np.nonzero(
            ~np.all(g["hybrid_solutions"] == np.zeros((50, 7)), axis=(1, 2))
        )[0]
        N = len(hybrid_indices)
        print(f"Found {N} hybrid demonstrations")
        with h5py.File(output_file, "w-") as f:
            for k in g.keys():
                f.create_dataset(k, (N, *g[k].shape[1:]))
            for ii, jj in enumerate(tqdm(hybrid_indices)):
                for k in g.keys():
                    f[k][ii, ...] = g[k][jj, ...]


def downsize_and_split(
    input_file: str, output_dir: str, train_size: int, val_size: int, test_size: int
):
    """
    This function is meant to be used to regularize the sizes of individual problem types
    (e.g. merged cubbies without neutral poses). Use this function on the output of either
    `merge_data_pipeline_files` or `extract_hybrid_expert_data` (depending on whether you
    want all the problems with global expert solutions or the subset that have hybrid expert solutions
    as well). This function will create three datasets, a train, val, and test dataset.
    If any of the sizes are set to 0, it will ignore that dataset

    :param input_file str: The input file, should come from one of the functions described above
    :param output_dir str: The output directory (this directory should exist but doesn't
                           need any subdirectories)
    :param train_size int: The size of the training dataset
    :param val_size int: The size of the validation dataset
    :param test_size int: The size of the test dataset
    """
    with h5py.File(input_file) as f:
        assert train_size + val_size + test_size < len(f["cuboid_centers"])
        indices = np.random.choice(
            np.arange(len(f["cuboid_centers"])),
            size=train_size + test_size + val_size,
            replace=False,
        )
        train_indices, val_indices, test_indices = (
            np.sort(indices[:train_size]),
            np.sort(indices[train_size : train_size + val_size]),
            np.sort(indices[train_size + val_size :]),
        )

        assert (
            len(train_indices) + len(val_indices) + len(test_indices)
            == train_size + val_size + test_size
        )

        path = Path(output_dir).resolve()

        if val_size > 0:
            (path / "val").mkdir(parents=True, exist_ok=True)
            with h5py.File(str(path / "val" / "val.hdf5"), "w-") as g:
                for k in f.keys():
                    g.create_dataset(k, (val_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(val_indices)):
                    for k in g.keys():
                        g[k][ii, ...] = f[k][jj, ...]
        if test_size > 0:
            (path / "test").mkdir(parents=True, exist_ok=True)
            with h5py.File(str(path / "test" / "test.hdf5"), "w-") as g:
                for k in f.keys():
                    g.create_dataset(k, (test_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(test_indices)):
                    for k in g.keys():
                        g[k][ii, ...] = f[k][jj, ...]
        if train_size > 0:
            (path / "train").mkdir(parents=True, exist_ok=True)
            with h5py.File(str(path / "train" / "train.hdf5"), "w-") as g:
                for k in f.keys():
                    g.create_dataset(k, (train_size, *f[k].shape[1:]))
                for ii, jj in enumerate(tqdm(train_indices)):
                    for k in g.keys():
                        g[k][ii, ...] = f[k][jj, ...]


def merge_scenes(input_dir: str, output_dir: str):
    """
    Merges all the `train.hdf5` files, merges all the `val.hdf5` files, and merges all the
    `test.hdf5` files to create three files that can be used to train. Use this function
    to merge datasets with different scene types and problem paradigms

    :param input_dir str: The directory containing all the datasets. This should have the
                          following structure (scene types are just examples)
                          input_dir/
                            dresser/
                              neutral/
                                train/
                                  train.hdf5
                                val/
                                  val.hdf5
                                test/
                                  test.hdf5
                              task_oriented/
                                train/
                                  train.hdf5
                                val/
                                  val.hdf5
                                test/
                                  test.hdf5
                            cubby/
                              ...
                            tabletop/
                              ...
    :param output_dir str: The final output directory where the data will live. This
                           directory should exist, but can be empty.
    """
    train_files = list(Path(input_dir).rglob("*train/train.hdf5"))
    if len(train_files) > 0:
        (Path(output_dir) / "train").mkdir(parents=True, exist_ok=True)
        merge_files(train_files, str(Path(output_dir) / "train" / "train.hdf5"))
    val_files = list(Path(input_dir).rglob("*val/val.hdf5"))
    if len(val_files) > 0:
        (Path(output_dir) / "val").mkdir(parents=True, exist_ok=True)
        merge_files(val_files, str(Path(output_dir) / "val" / "val.hdf5"))
    test_files = list(Path(input_dir).rglob("*test/test.hdf5"))
    if len(test_files) > 0:
        (Path(output_dir) / "test").mkdir(parents=True, exist_ok=True)
        merge_files(test_files, str(Path(output_dir) / "test" / "test.hdf5"))


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        help="What stage of the data pipeline to run", dest="stage"
    )
    merge_single_parser = subparsers.add_parser(
        "merge-single-scene",
        help=(
            "Will merge a bunch of final output files from the `gen_data.py` script."
            " These files should all correspond to the same problem paradigm"
            " (i.e. with or without neutral poses) of the same scene type"
            " (i.e. dresser, cubby, merged cubby, or tabletop). For example, they"
            " should all be dresser scenes going to/from a neutral pose."
        ),
    )
    merge_single_parser.add_argument(
        "data_dir",
        type=str,
        help=(
            "A directory containing all the merged files (the final outputs)"
            " of the `gen_data.py` script. Note that the script's final output"
            " is always named `all_data.hdf5`, so these files should either be"
            " renamed or kept in unique folders that live within `data_path`."
            " This function will do a recursive search within this folder to find"
            " all `*.hdf5` files."
        ),
    )
    merge_single_parser.add_argument(
        "output_file",
        type=str,
        help="The file to output. Should be something like `blahblahblah.hdf5`",
    )

    hybrid_parser = subparsers.add_parser(
        "extract-hybrid",
        help=(
            "Takes a file generated by `merge_data_pipeline_files` and removes all trajectories"
            " that do not have a hybrid expert solution. This is necessary if you want to train on"
            " only hybrid solutions because it's much faster to do this once during processing than"
            " to rely on the dataloader to do it."
        ),
    )
    hybrid_parser.add_argument(
        "input_file",
        type=str,
        help="The output of the merge-single-scene phase. Should be something like `blahblahblah.hdf5`",
    )
    hybrid_parser.add_argument(
        "output_file",
        type=str,
        help=(
            "The file to output with only the problems with hybrid expert"
            " demonstrations. Should be something like `somethingsomething.hdf5`"
        ),
    )

    downsize_parser = subparsers.add_parser(
        "downsize-and-split",
        help=(
            "This function is meant to be used to regularize the sizes of individual problem types"
            " (e.g. merged cubbies without neutral poses). Use this function on the output of either"
            " `merge_data_pipeline_files` or `extract_hybrid_expert_data` (depending on whether you"
            " want all the problems with global expert solutions or the subset that have hybrid expert solutions"
            " as well). This function will create three datasets, a train, val, and test dataset."
            " If any of the sizes are set to 0, it will ignore that dataset"
        ),
    )
    downsize_parser.add_argument(
        "input_file",
        type=str,
        help="The input file, should come from one of the functions described above",
    )
    downsize_parser.add_argument(
        "output_dir",
        type=str,
        help=(
            "The output directory (this directory should exist but doesn't"
            " need any subdirectories)"
        ),
    )
    downsize_parser.add_argument(
        "train_size",
        type=int,
        help="The size of the training dataset (must be less than the full dataset)",
    )
    downsize_parser.add_argument(
        "val_size",
        type=int,
        help="The size of the validation dataset (must be less than the full dataset)",
    )
    downsize_parser.add_argument(
        "test_size",
        type=int,
        help="The size of the test dataset (must be less than the full dataset)",
    )

    merge_all_parser = subparsers.add_parser(
        "final-merge",
        formatter_class=RawTextHelpFormatter,
        help=(
            "Merges all the `train.hdf5` files, merges all the `val.hdf5` files, and merges all the"
            " `test.hdf5` files to create three files that can be used to train. Use this function"
            " to merge datasets with different scene types and problem paradigms"
        ),
    )
    merge_all_parser.add_argument(
        "input_dir",
        type=str,
        help=(
            "The directory containing all the datasets. This should have the"
            " following structure (scene types are just examples)\n"
            "input_dir/\n"
            "  dresser/\n"
            "      neutral/\n"
            "        train/\n"
            "          train.hdf5\n"
            "        val/\n"
            "          val.hdf5\n"
            "        test/\n"
            "          test.hdf5\n"
            "     task_oriented/\n"
            "        train/\n"
            "          train.hdf5\n"
            "        val/\n"
            "          val.hdf5\n"
            "        test/\n"
            "          test.hdf5\n"
            "    cubby/\n"
            "      ...\n"
            "    tabletop/\n"
            "      ...\n"
        ),
    )
    merge_all_parser.add_argument(
        "output_dir",
        type=str,
        help=(
            "The final output directory where the data will live. This"
            " directory should exist, but can be empty."
        ),
    )

    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.stage == "merge-single-scene":
        merge_data_pipeline_files(args.data_dir, args.output_file)
    elif args.stage == "extract-hybrid":
        extract_hybrid_expert_data(args.input_file, args.output_file)
    elif args.stage == "downsize-and-split":
        downsize_and_split(
            args.input_file,
            args.output_dir,
            args.train_size,
            args.val_size,
            args.test_size,
        )
    elif args.stage == "final-merge":
        merge_scenes(args.input_dir, args.output_dir)
