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


def train(model, train_loader, optimizer, epochs, checkpoints=0, log_dir='.',
          scheduler=None, val_set=None, estimator=None, evaluator=None):
    """
    Implements the training loop for an experiment.

    Parameters
    ----------
    model : TranscriptionModel
      Model to train
    train_loader : DataLoader
      PyTorch Dataloader object for retrieving batches of data
    optimizer : Optimizer
      PyTorch Optimizer for updating weights
    epochs : int
      Number of loops through the dataset;
      Each loop contains a snippet of each track exactly once
    checkpoints : int
      Number of batches in between checkpoints
    log_dir : str
      Path to directory for saving model, optimizer state, and events
    scheduler : Scheduler or None (optional)
      PyTorch Scheduler used to update learning rate
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

    # Initialize a writer to log any reported results
    writer = SummaryWriter(log_dir)

    # Start at iteration 0 by default
    start_iter = 0

    # Make sure the model is in training mode
    model.train()

    for epoch in tqdm(range(start_iter, epochs)):
        # Collection of losses for each batch in the loop
        train_loss = dict()
        # Loop through the dataset
        for batch in tqdm(train_loader, desc='Step'):
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
                # Save the model checkpoint
                torch.save(model, os.path.join(log_dir, f'{tools.PYT_MODEL}-{model.iter}.{tools.PYT_EXT}'))
                # Save the optimizer state at the checkpoint
                torch.save(optimizer.state_dict(), os.path.join(log_dir, f'{tools.PYT_STATE}-{model.iter}.{tools.PYT_EXT}'))

                if val_set is not None and evaluator is not None:
                    # Validate the current model weights
                    validate(model, val_set, evaluator, estimator)
                    # Average the results, log them, and reset the tracking
                    evaluator.finalize(writer, model.iter)
                    # TODO - add forced stopping criterion?
                    # Make sure the model is back in training mode
                    model.train()

        if scheduler is not None:
            # Perform a learning rate scheduler step
            scheduler.step()

    # Save the final model
    torch.save(model, os.path.join(log_dir, f'{tools.PYT_MODEL}-{model.iter}.{tools.PYT_EXT}'))
    # Save the final optimizer state
    torch.save(optimizer.state_dict(), os.path.join(log_dir, f'{tools.PYT_STATE}-{model.iter}.{tools.PYT_EXT}'))

    if val_set is not None and evaluator is not None:
        # Validate the current model weights
        validate(model, val_set, evaluator, estimator)
        # Average the results, log them, and reset the tracking
        evaluator.finalize(writer, model.iter)

    return model
