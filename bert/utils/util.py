import os
import pickle
import torch
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

try:
    from transformers.trainer_utils import TrainerControl
except ImportError:
    class TrainerControl:
        def __init__(self):
            self.should_save = False
            self.should_evaluate = False
            self.should_log = False
            self.should_epoch_stop = False
            self.should_training_stop = False

from transformers import Trainer, TrainerCallback

def compute_metrics(eval_pred):
    """
    Compute accuracy for masked language modeling.
    Accuracy is computed only on positions where the label is not -100.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if np.sum(mask) == 0:
        accuracy = 0.0
    else:
        accuracy = (predictions[mask] == labels[mask]).mean()
    return {'eval_accuracy': accuracy}


class EmptyCacheCallback(TrainerCallback):
    """
    A callback that clears the CUDA cache at the end of each training step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control


class DiskFlushTrainer(Trainer):
    """
    A custom Trainer subclass that, during evaluation, flushes intermediate predictions
    and labels to disk when GPU memory usage exceeds a threshold.
    The evaluation loop is wrapped with tqdm to display a progress bar.
    """
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        # Create a temporary directory for flush files.
        temp_dir = "./temp_eval_results"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get evaluation DataLoader.
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        # Determine GPU memory threshold.
        device = self.args.device
        device_props = torch.cuda.get_device_properties(device)
        total_mem = device_props.total_memory
        flush_threshold = 0.95  # 95% usage threshold
        
        # Lists for accumulating predictions and labels.
        all_preds = []
        all_labels = []
        flush_files = []
        flush_count = 0
        
        self.model.eval()
        # Wrap evaluation loop with tqdm to show progress.
        for step, inputs in tqdm(
            enumerate(eval_dataloader),
            total=len(eval_dataloader),
            desc="Evaluating",
            leave=False
        ):
            with torch.no_grad():
                outputs = self.prediction_step(
                    self.model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
                )
            # outputs is a tuple: (loss, logits, labels)
            logits = outputs[1]
            labels = outputs[2]
            
            if logits is not None:
                logits = logits.detach().cpu()
            if labels is not None:
                labels = labels.detach().cpu()
                
            if logits is not None:
                all_preds.append(logits)
            if labels is not None:
                all_labels.append(labels)
            
            # Check current GPU memory usage.
            current_mem = torch.cuda.memory_allocated(device)
            if current_mem / total_mem >= flush_threshold:
                # Flush the accumulated predictions and labels to disk.
                file_path = os.path.join(temp_dir, f"eval_flush_{flush_count}.pkl")
                with open(file_path, "wb") as f:
                    pickle.dump((all_preds, all_labels), f)
                flush_files.append(file_path)
                flush_count += 1
                # Clear the in-memory lists.
                all_preds = []
                all_labels = []
                torch.cuda.empty_cache()
        
        # Flush any remaining results.
        if all_preds or all_labels:
            file_path = os.path.join(temp_dir, f"eval_flush_{flush_count}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump((all_preds, all_labels), f)
            flush_files.append(file_path)
        
        # Read and combine all flushed files.
        combined_preds = []
        combined_labels = []
        for file_path in flush_files:
            with open(file_path, "rb") as f:
                preds_list, labels_list = pickle.load(f)
            if preds_list:
                combined_preds.extend(preds_list)
            if labels_list:
                combined_labels.extend(labels_list)
            os.remove(file_path)
        # Clean up the temporary directory if empty.
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        
        # Concatenate all predictions and labels.
        if combined_preds:
            combined_preds = torch.cat(combined_preds, dim=0)
        else:
            combined_preds = None
        if combined_labels:
            combined_labels = torch.cat(combined_labels, dim=0)
        else:
            combined_labels = None
        
        # Compute metrics using the provided compute_metrics function.
        metrics = {}
        if self.compute_metrics is not None and combined_preds is not None and combined_labels is not None:
            metrics = self.compute_metrics((combined_preds.numpy(), combined_labels.numpy()))
        # Reset Trainer control.
        self.control = TrainerControl()
        return metrics


class SaveBestModelCallback(TrainerCallback):
    """
    A callback that monitors the evaluation metric and, if a new best model is found,
    saves the model's state dictionary to a file.
    If no output directory is provided in the callback, it uses args.output_dir.
    The file is named with the current timestamp (e.g. "best_model_20230426_153045.pth").
    """
    def __init__(self, monitor: str = "eval_accuracy", output_dir=None):
        self.monitor = monitor
        self.best_metric = None
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.monitor not in metrics:
            return control

        current_metric = metrics[self.monitor]
        # Save model if this evaluation metric is better than the previous best.
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_dir if self.output_dir is not None else args.output_dir
            save_path = os.path.join(output_dir, f"best_model_{timestamp}.pth")
            torch.save(kwargs["model"].state_dict(), save_path)
            print(f"New best model saved to {save_path} with {self.monitor}: {current_metric}")
        return control
