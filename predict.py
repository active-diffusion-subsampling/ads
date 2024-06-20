from keras.src import backend


def get_predict_on_batch_fn():
    """
    returns a backend-specific predict_on_batch function with the following signature:
        predict_on_batch(trainer: Trainer, input: Tensor) -> output: Tensor

    This works the same as Trainer.predict_on_batch but without mapping the output
    tensors to numpy arrays, which facilitates gradient tracking through predict_on_batch.
    """
    if backend.backend() == "jax":
        return predict_on_batch_jax
    elif backend.backend() == "tensorflow":
        return predict_on_batch_tf
    elif backend.backend() == "torch":
        return predict_on_batch_torch
    else:
        raise ValueError(f"Backend {backend.backend()} not supported")


def predict_on_batch_jax(trainer, x):
    if not all(layer.built for layer in trainer._flatten_layers()):
        # Build model
        with backend.StatelessScope():
            trainer(x)
    trainer._record_training_state_sharding_spec()
    trainer.make_predict_function()

    trainable_variables = [v.value for v in trainer.trainable_variables]
    non_trainable_variables = [v.value for v in trainer.non_trainable_variables]
    state = (trainable_variables, non_trainable_variables)
    batch_outputs, state = trainer.predict_function(state, [(x,)])
    return batch_outputs


def predict_on_batch_torch(trainer, x):
    trainer.make_predict_function()
    batch_outputs = trainer.predict_function([(x,)])
    return batch_outputs


def predict_on_batch_tf(trainer, x):
    trainer.make_predict_function()
    batch_outputs = trainer.predict_function([(x,)])
    return batch_outputs
