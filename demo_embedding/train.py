# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.evaluate import validate, append_results, average_results, log_results
from amt_tools import tools

# Regular imports
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import os

__all__ = [
    'train'
]


def train(model, train_loader, optimizer, epochs, checkpoints=0, log_dir='.', scheduler=None,
          resume=True, val_set=None, estimator=None, evaluator=None):
    """
    Implements the training loop for an experiment.

    Parameters
    ----------
    model : TranscriptionModel
      Model to train
    train_loader : DataLoader
      PyTorch Dataloader object for retrieving batches of data
    optimizer : Optimizer
      PyTorch Optimizer for updating weights - expected to have only one parameter group
    epochs : int
      Number of loops through the dataset;
      Each loop may be comprised of multiple batches;
      Each loop contains a snippet of each track exactly once
    checkpoints : int
      Number of equally spaced save/validation checkpoints - 0 to disable
    log_dir : str
      Path to directory for saving model, optimizer state, and events
    scheduler : Scheduler or None (optional)
      PyTorch Scheduler used to update learning rate
    resume : bool
      Start from most recently saved model and optimizer state
    val_set : TranscriptionDataset or None (optional)
      Dataset to use for validation loops
    estimator : Estimator
      Estimation protocol to use during validation
    evaluator : Evaluator
      Evaluation protocol to use during validation

    Returns
    ----------
    model : TranscriptionModel
      Trained model
    """

    # TODO - multi_gpu - DataParallel removes ability to call .run_on_batch()
    #                  - Can DataParallel be hooked into TranscriptionModel to
    #                    only call the forward() function?

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # Start at iteration 0 by default
    start_iter = 0

    if resume:
        # Obtain the files that already exist in the log directory
        log_files = os.listdir(log_dir)

        # Extract and sort files pertaining to the model
        model_files = sorted([path for path in log_files if tools.PYT_MODEL in path], key=tools.file_sort)
        # Extract and sort files pertaining to the optimizer state
        optimizer_files = sorted([path for path in log_files if tools.PYT_STATE in path], key=tools.file_sort)

        # Check if any previous checkpoints exist
        if len(model_files) > 0 and len(optimizer_files) > 0:
            # Define the tags for the model and state at the selected number of iterations
            max_model = f'{tools.PYT_MODEL}-{epochs}.{tools.PYT_EXT}'
            max_state = f'{tools.PYT_STATE}-{epochs}.{tools.PYT_EXT}'

            # Check if a model/state already exists for the selected iterations
            model_index = model_files.index(max_model) if max_model in model_files else -1
            state_index = optimizer_files.index(max_state) if max_state in optimizer_files else -1

            # Get the path to the latest model file
            model_path = os.path.join(log_dir, model_files[model_index])
            # Get the path to the latest optimizer state file
            optimizer_path = os.path.join(log_dir, optimizer_files[state_index])

            # Make the start iteration the iteration when the checkpoint was taken
            start_iter = int(''.join([ch for ch in model_files[-1] if ch.isdigit()]))
            # Get the iteration for the latest optimizer state
            optimizer_iter = int(''.join([ch for ch in optimizer_files[-1] if ch.isdigit()]))
            # Make sure these iterations match
            assert start_iter == optimizer_iter

            # Determine the device to use
            device = model.device
            # Load the latest model and replace the parameterized version
            model = torch.load(model_path, map_location=device)
            model.change_device(device)
            model.train()
            # Replace the randomly initialized parameters with the saved parameters
            # TODO - allow for saving/loading of optimizer with multiple parameter groups
            super(type(optimizer), optimizer).__init__(model.parameters(), optimizer.defaults)
            # Load the latest optimizer state into the parameterized version
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))

    # Make sure the model is in training mode
    model.train()

    for epoch in tqdm(range(start_iter, epochs)):
        # Collection of losses for each batch in the loop
        train_loss = dict()
        # Loop through the dataset
        for batch in train_loader:
            # batch["audio"] = batch["audio"].to(model.device)
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the predictions and loss for the batch
            preds = model.run_on_batch(batch)
            # Extract the loss from the output
            batch_loss = preds[tools.KEY_LOSS]
            # Compute gradients based on total loss
            batch_loss[tools.KEY_LOSS_TOTAL].backward()
            # Add all of the losses to the collection
            train_loss = append_results(train_loss, tools.dict_to_array(batch_loss))
            # Perform an optimization step
            optimizer.step()

            # Average the loss from all of the batches within this loop
            train_loss = average_results(train_loss)
            # Log the training loss(es)
            log_results(train_loss, writer, step=model.iter, tag=f'{tools.TRAIN}/{tools.KEY_LOSS}')

            # Increase the iteration count by one
            model.iter += 1

            if model.iter % checkpoints == 0:
                # Save the model
                torch.save(model, os.path.join(log_dir, f'{tools.PYT_MODEL}-{model.iter}.{tools.PYT_EXT}'))
                # Save the optimizer sate
                torch.save(optimizer.state_dict(), os.path.join(log_dir, f'{tools.PYT_STATE}-{model.iter}.{tools.PYT_EXT}'))

                # If we are at a checkpoint, and a validation set with an estimator is available
                if val_set is not None and evaluator is not None:
                    # Validate the current model weights
                    validate(model, val_set, evaluator, estimator)
                    # Average the results, log them, and reset the tracking
                    evaluator.finalize(writer, model.iter)
                    # Make sure the model is back in training mode
                    model.train()

        if scheduler is not None:
            # Perform a learning rate scheduler step
            scheduler.step()

    # Save the model
    torch.save(model, os.path.join(log_dir, f'{tools.PYT_MODEL}-{model.iter}.{tools.PYT_EXT}'))
    # Save the optimizer sate
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'{tools.PYT_STATE}-{model.iter}.{tools.PYT_EXT}'))

    # If we are at a checkpoint, and a validation set with an estimator is available
    if val_set is not None and evaluator is not None:
        # Validate the current model weights
        validate(model, val_set, evaluator, estimator)
        # Average the results, log them, and reset the tracking
        evaluator.finalize(writer, model.iter)

    return model
