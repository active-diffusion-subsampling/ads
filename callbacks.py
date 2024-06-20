from pathlib import Path

import wandb
from keras.callbacks import Callback

from utils.keras_utils import sample_images


class DDIMSamplingCallback(Callback):
    """Sample and plot random generated images for visual evaluation of generation quality"""

    def __init__(
        self,
        diffusion_model,
        image_shape,
        diffusion_steps,
        batch_size,
        save_dir,
        postprocess_func,
        n_frames=1,
        start_with_eval=False,
        wandb_log=False,
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.image_shape = image_shape
        self.diffusion_steps = diffusion_steps
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.n_frames = n_frames
        self.postprocess_func = postprocess_func
        self.start_with_eval = start_with_eval
        self.wandb_log = wandb_log

        self.save_dir.mkdir(exist_ok=True)

        self.seed = 42

    def on_epoch_end(self, epoch, logs=None):
        file_path = Path(self.save_dir) / f"samples_epoch_{epoch}.png"
        image_shape = [self.batch_size, *self.image_shape]
        print("\n")
        sample_images(
            self.diffusion_model,
            image_shape,
            self.diffusion_steps,
            file_path,
            n_frames=self.n_frames,
            postprocess_func=self.postprocess_func,
            animate_diffusion_process=False,
            seed=self.seed,
        )
        if self.wandb_log:
            if self.n_frames == 1:
                file_path = str(file_path).replace(".png", "_generated.png")
                wandb.log(
                    {
                        "DDIM-samples": [
                            wandb.Image(file_path),
                        ]
                    }
                )
            else:
                file_path = str(file_path).replace(".png", "_generated.gif")
                wandb.log(
                    {
                        "DDIM-samples": [
                            wandb.Video(
                                file_path,
                            )
                        ]
                    }
                )

    def on_train_begin(self, logs=None):
        if not self.start_with_eval:
            return
        return self.on_epoch_end(0, logs)
