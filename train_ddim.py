"""
Training DDIM model
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["MPLBACKEND"] = "Agg"

import keras
import tensorflow as tf
import utils.lib.utils as lib_utils
import yaml
from callbacks import DDIMSamplingCallback
from keras import ops
from utils.lib.config import Config, load_config_from_yaml

from datasets import get_datasets

from ddim import DiffusionModel
from utils.keras_utils import ClearMemory, get_loss_func, get_postprocess_fn, plot_batch


def print_train_summary(config):
    """Print training summary from config."""
    print("=" * 57)
    print("Training Summary:")
    print("=" * 57)
    print(f"| {'Parameter':<20} | {'Value':<30} |")
    print("|" + "-" * 55 + "|")
    print(f"| {'Epochs':<20} | {config.optimization.num_epochs:<30} |")
    print(f"| {'Learning rate':<20} | {config.optimization.learning_rate:<30} |")
    print(
        f"| {'Latent diffusion':<20} | {'ON' if config.model.latent else 'OFF':<30} |"
    )
    print(
        f"| {'Image shape':<20} | {', '.join(str(dim) for dim in config.data.image_shape_after_augmentations):<30} |"
    )
    if config.model.latent:
        if config.model.get("latent_shape") is not None:
            print(
                f"| {'Latent shape':<20} | "
                f"{', '.join(str(dim) for dim in config.model.latent_shape):<30} |"
            )
    print(
        f"| {'Normalized range':<20} | "
        f"{', '.join(str(val) for val in config.data.normalization):<30} |"
    )
    print(f"| {'#Frames':<20} | {config.data.n_frames:<30} |")
    if config.data.extension == "tf_dataset":
        dataset = f"{config.data.dataset_folder}/{config.data.dataset_name}/{config.data.dataset_version}"
    else:
        dataset = config.data.train_folder

    if isinstance(dataset, list):
        dataset = [Path(folder).name for folder in dataset]
        dataset = ",".join(dataset)
    else:
        if config.data.extension == "tf_dataset":
            dataset = ":".join(dataset.split("/")[-2:])
        else:
            dataset = Path(dataset).name
    print(f"| {'Dataset':<20} | {dataset:<30} |")
    print("=" * 57)


def train_ddim(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    config: Config,
    run_dir: str,
    postprocess_func: callable,
    train: bool = True,
) -> DiffusionModel:
    """
    Trains the DDIM (Diffusion Model) using the provided datasets and configuration.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.
        val_dataset (tf.data.Dataset): The validation dataset.
        config (Config): The configuration object.
        run_dir (str): The directory to save the training results.
        postprocess_func (callable): The postprocessing function to apply to the generated images
            such that they can be visualized.
        train (bool, optional): Whether to perform training or not. Defaults to True.

    Returns:
        DiffusionModel: The trained DDIM model.
    """
    weight_decay = config.optimization.weight_decay
    learning_rate = config.optimization.learning_rate
    num_epochs = config.optimization.num_epochs
    widths = config.model.widths
    block_depth = config.model.block_depth
    ema_val = config.optimization.ema
    min_signal_rate = config.sampling.min_signal_rate
    max_signal_rate = config.sampling.max_signal_rate
    diffusion_steps = config.model.diffusion_steps
    compute_kid = config.evaluation.kid.enable
    kid_diffusion_steps = config.evaluation.kid.diffusion_steps
    kid_image_shape = config.evaluation.kid.image_shape
    run_eagerly = config.model.run_eagerly
    latent = config.model.latent

    image_shape = train_dataset.element_spec.shape[1:].as_list()

    # if specified in config, check if image shape from dataset matches the specified image shape
    specified_image_shape = config.data.image_shape_after_augmentations
    if specified_image_shape:
        assert image_shape == specified_image_shape[:-1] + [
            specified_image_shape[-1] * config.data.n_frames
        ], (
            f"Image shape from dataset {image_shape} does not match image shape "
            f"specified in config {specified_image_shape} with n_frames {config.data.n_frames}."
        )
    else:
        config.data.image_shape_after_augmentations = image_shape[:-1] + [
            image_shape[-1] // config.data.n_frames
        ]

    run_eagerly = config.model.run_eagerly if not sys.gettrace() else True

    # create and compile the model
    model = DiffusionModel(
        image_shape,
        widths,
        block_depth,
        ema_val=ema_val,
        min_signal_rate=min_signal_rate,
        max_signal_rate=max_signal_rate,
        diffusion_steps=diffusion_steps,
        image_range=config.data.normalization,
        compute_kid=compute_kid,
        kid_diffusion_steps=kid_diffusion_steps,
        kid_image_shape=kid_image_shape,
        # mean=config.data.mean,
        # variance=config.data.variance,
        latent_diffusion=latent,
        latent_shape=config.model.latent_shape if latent else None,
        autoencoder_checkpoint_directory=config.model.get(
            "autoencoder_checkpoint_directory"
        ),
    )

    checkpoint_path = Path(run_dir) / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    # save model architecture to run_dir/checkpoints/model.json
    model.save_model_json(checkpoint_path)

    loss_func = get_loss_func(config.optimization.loss)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=loss_func,
        run_eagerly=run_eagerly,
    )  # pixelwise mean absolute error is used as loss

    # # save the best model based on the validation KID metric
    checkpoint_file = str(checkpoint_path / "diffusion_model_{epoch}.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        monitor="i_loss",
        mode="min",
        save_best_only=False,
    )

    validation_sample_callback = DDIMSamplingCallback(
        model,
        image_shape,
        config.evaluation.diffusion_steps,
        config.evaluation.batch_size,
        save_dir=run_dir / "samples",
        n_frames=config.data.n_frames,
        postprocess_func=postprocess_func,
        start_with_eval=config.evaluation.get("start_with_eval", False),
    )

    # save config to yaml file without reordering in run_dir
    config.save_to_yaml(Path(run_dir) / "config.yaml")

    print_train_summary(config)

    callbacks = [
        validation_sample_callback,
        checkpoint_callback,
        ClearMemory(),
    ]

    # run training and plot generated images periodically
    if train:
        # weird bug where i need to reinitialize the datasets
        train_dataset, val_dataset = get_datasets(config.data.user.data_root, config)

        start_training_str = (
            f"Starting training for {num_epochs} "
            f"epochs on {lib_utils.get_date_string()}..."
        )
        print("-" * len(start_training_str))
        print(start_training_str)
        model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=config.optimization.get("steps_per_epoch"),
        )
    return model


def parse_args():
    """Parse arguments for training DDIM."""
    parser = argparse.ArgumentParser(description="DDIM training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/training/ddim_train_fastmri.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-d",
        "--data_root",
        type=str,
        default="data/",
        help="Path to the your data directory.",
    )
    parser.add_argument(
        "-r",
        "--run_dir",
        type=str,
        default="trained_models/",
        help="Base path of the running directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # load config from yaml file
    args = parse_args()
    config = load_config_from_yaml(Path(args.config), loader=yaml.UnsafeLoader)
    data_root = args.data_root
    config.data.__setattr__("user", {"data_root": data_root})

    print(f"\n🔔 You are using data from {data_root}\n")

    keras.utils.set_random_seed(config.seed)

    train_dataset, val_dataset = get_datasets(data_root, config)

    date = lib_utils.get_date_string()
    debug_str = "_debug" if sys.gettrace() else ""
    run_dir = Path(args.run_dir) / (date + "_" + Path(args.config).stem + debug_str)
    run_dir.mkdir(exist_ok=True, parents=True)
    config.run_dir = str(run_dir)

    train_batch = list(train_dataset.take(1))[0]
    val_batch = list(val_dataset.take(1))[0]

    postprocess_func = get_postprocess_fn(config)

    if config.data.n_frames == 1:
        batch_plot_path = run_dir / "images" / "batch.png"
        plot_batch(
            postprocess_func(train_batch),
            postprocess_func(val_batch),
            batch_plot_path,
            aspect="auto",
        )
    else:
        raise UserWarning("Multi-frame data is not supported")

    model = train_ddim(
        train_dataset,
        val_dataset,
        config,
        run_dir,
        postprocess_func,
        train=True,
    )
    print(f"Training complete. Check results and cpkt in {run_dir}")
