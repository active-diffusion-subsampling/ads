import json
from pathlib import Path

import keras
from utils.lib import log

from models.stable_diffusion import StableDiffusion
from utils.keras_utils import get_latest_checkpoint_path


def load_model(
    directory,
    checkpoint_file=None,
    verbose=False,
    stable_diffusion_kwargs=None,
    **kwargs,
):
    """Load model from run_dir."""
    if "stable_diffusion" in str(directory):
        model = StableDiffusion(**stable_diffusion_kwargs)
        return model

    json_path = str(Path(directory) / "model.json")
    with open(json_path, "r", encoding="utf-8") as file:
        json_data = json.loads(file.read())
    # if **kwargs than update json_data with kwargs
    json_data["config"].update(kwargs)
    model_name = json_data["config"]["name"]

    # this is how we save the weights for the models currently
    if json_data["module"] == "vae":
        json_data["module"] = "vae_model"
        log.warning(
            "Updated module name from 'vae' to 'vae_model' in outdated model.json file. "
            "Please update the model.json file."
        )

    # take json_data["config"]["autoencoder_checkpoint_directory"] relative to directory
    # from "pretrained" folder onwards
    if (
        "autoencoder_checkpoint_directory" in json_data["config"]
        and json_data["config"]["autoencoder_checkpoint_directory"] is not None
    ):

        autoencoder_checkpoint_directory = json_data["config"][
            "autoencoder_checkpoint_directory"
        ]
        # split at pretrained folder to get the correct path
        autoencoder_checkpoint_directory = autoencoder_checkpoint_directory.split(
            "pretrained/"
        )[1]
        json_data["config"]["autoencoder_checkpoint_directory"] = (
            Path(str(directory).split("pretrained/", maxsplit=1)[0])
            / "pretrained"
            / autoencoder_checkpoint_directory
        )
        assert json_data["config"]["autoencoder_checkpoint_directory"].is_dir(), (
            f"Autoencoder checkpoint directory {json_data['config']['autoencoder_checkpoint_directory']} "
            f"does not exist, please update `config.model.autoencoder_checkpoint_directory` in {directory}/model.json"
        )
        json_data["config"]["autoencoder_checkpoint_directory"] = str(
            json_data["config"]["autoencoder_checkpoint_directory"]
        )

    # convert back to json string
    json_string = json.dumps(json_data)
    model = keras.models.model_from_json(json_string)

    if checkpoint_file is None:
        checkpoint_path = get_latest_checkpoint_path(directory, model_name=model_name)
    else:
        checkpoint_path = Path(directory) / checkpoint_file

    try:
        try:
            model.load_weights(str(checkpoint_path))
        except:
            model.load_weights(str(checkpoint_path), skip_mismatch=True)
            log.warning("Loaded weights with some mismatches")

    except Exception as e:
        raise ValueError(f"Could not load weights from {checkpoint_path}\n{e}") from e
    if verbose:
        model.summary()

    model_name = json_data["class_name"]
    log.success(
        f"Succesfully loaded {model_name} with weights from {log.yellow(checkpoint_path)}"
    )
    return model
