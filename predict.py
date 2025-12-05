import os  # Provides functions for interacting with the operating system (paths, directories, etc.)
import argparse  # Used to parse command-line arguments when the script is executed
import re  # Regular expressions, useful for parsing file names and extracting timestamps
import datetime as dt  # Convenient alias for working with dates and times

import torch  # PyTorch library, used here for loading the trained model and running inference

from rainpred.model import RainPredRNN  # The neural network architecture used for rain prediction
from rainpred.geo_io import load_sequence_from_dir, save_predictions_as_geotiff  # Helpers for reading/writing GeoTIFF radar data


# Regular expression used to parse file names of the form:
#   rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
# or any similar pattern where:
#   - a prefix ends with an underscore "_"
#   - then an 8-digit date YYYYMMDD
#   - then the letter "Z"
#   - then a 4-digit time hhmm
#   - then an arbitrary suffix (e.g. "_VMI.tiff")
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+_)(?P<date>\d{8})Z(?P<hour>\d{2})(?P<minute>\d{2})(?P<suffix>.*)$"
)  # Named groups make it easy to rebuild the file name later


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Decide which device (CPU or GPU) should be used.

    Parameters
    ----------
    force_cpu : bool
        When True, always use the CPU even if a CUDA GPU is available.

    Returns
    -------
    torch.device
        The selected device: "cuda" if available and not forced to CPU, otherwise "cpu".
    """
    if (not force_cpu) and torch.cuda.is_available():  # Check if CUDA is available and CPU is not forced
        return torch.device("cuda")  # Use the first CUDA device
    return torch.device("cpu")  # Fallback: use CPU


def load_model(checkpoint_path: str, device: torch.device, input_length: int, pred_length: int) -> torch.nn.Module:
    """
    Load a trained RainPredRNN model from a checkpoint file.

    This function is written to be robust against different ways the checkpoint
    might have been saved (entire model object vs. state_dict).

    The important fix for PyTorch >= 2.6 is that we explicitly set
    `weights_only=False` in torch.load, which restores the old behavior
    and allows unpickling of Python objects (including NumPy arrays).

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint file (.pth or .pt).
    device : torch.device
        Device where the model will be moved (CPU or CUDA).
    input_length : int
        Number of input frames the model expects (m).
    pred_length : int
        Number of future frames to predict (n).

    Returns
    -------
    torch.nn.Module
        The loaded model, moved to the requested device and set to evaluation mode.
    """
    # IMPORTANT: in PyTorch 2.6 the default is weights_only=True.
    # Here we explicitly set weights_only=False because we trust our own checkpoint.
    ckpt = torch.load(
        checkpoint_path,  # Path to the checkpoint file
        map_location=device,  # Load tensors directly on the chosen device
        weights_only=False,  # Allow full unpickling (restores pre-2.6 behavior)
    )

    # Case 1: the checkpoint itself is already a PyTorch module (model was saved directly)
    if isinstance(ckpt, torch.nn.Module):  # Check if the loaded object is a Module instance
        model = ckpt  # Use it as the model
    # Case 2: the checkpoint is a dictionary, which is a very common pattern
    elif isinstance(ckpt, dict):  # When we saved multiple objects (e.g. model, optimizer, epoch, ...)
        # If a full model object was stored under "model"
        if "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):  # Check if a model key exists
            model = ckpt["model"]  # Extract the model
        # If we only stored the model parameters (state_dict)
        elif "model_state_dict" in ckpt:  # Typical key when saving model.state_dict()
            # Instantiate a fresh model with the known lengths
            model = RainPredRNN(input_length=input_length, pred_length=pred_length)  # Create model with given hyper-parameters
            model.load_state_dict(ckpt["model_state_dict"])  # Load the learned parameters
        elif "state_dict" in ckpt:  # Alternative naming convention
            model = RainPredRNN(input_length=input_length, pred_length=pred_length)  # Create model instance
            model.load_state_dict(ckpt["state_dict"])  # Load parameters
        else:
            # If we arrive here, we do not know how to extract the model from the checkpoint
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' is a dict but does not contain 'model', "
                f"'model_state_dict' or 'state_dict' keys. Available keys: {list(ckpt.keys())}"
            )
    else:
        # Any other object type is unexpected for a model checkpoint
        raise TypeError(
            f"Unsupported checkpoint type: {type(ckpt)}. "
            f"Expected torch.nn.Module or dict with 'model_state_dict' or 'model'."
        )

    model.to(device)  # Move the model to the selected device (CPU or GPU)
    model.eval()  # Put the model in evaluation mode (disables dropout, etc.)

    return model  # Return the ready-to-use model


def infer_time_step_minutes(file_list):
    """
    Infer the time step (in minutes) between consecutive frames from the file names.

    If at least two files match the expected pattern, we compute the difference
    in minutes between the last two timestamps. Otherwise we fall back to 10 minutes.

    Parameters
    ----------
    file_list : list of str
        List of file paths for the input radar frames.

    Returns
    -------
    int
        Time step in minutes between frames (default 10 if it cannot be inferred).
    """
    if len(file_list) < 2:  # We need at least two frames to infer a time step
        return 10  # Reasonable default for typical radar data (10-minute interval)

    # Parse the last two file names using the regular expression
    m2 = FILENAME_RE.match(os.path.basename(file_list[-1]))  # Newest frame
    m1 = FILENAME_RE.match(os.path.basename(file_list[-2]))  # Previous frame

    if not (m1 and m2):  # If either file name does not match the pattern
        return 10  # Fall back to a 10-minute step

    # Build datetime objects for both frames
    t1 = dt.datetime.strptime(m1.group("date") + m1.group("hour") + m1.group("minute"), "%Y%m%d%H%M")
    t2 = dt.datetime.strptime(m2.group("date") + m2.group("hour") + m2.group("minute"), "%Y%m%d%H%M")

    delta = t2 - t1  # Compute the time difference between the two timestamps
    minutes = int(delta.total_seconds() / 60)  # Convert the difference to minutes

    # Avoid returning zero or negative intervals; if that happens, fall back to 10.
    return minutes if minutes > 0 else 10  # Ensure a positive, non-zero time step


def build_output_basenames(file_list, n_future: int):
    """
    Build the base file names for the predicted frames.

    The idea is to:
      - Take the last input frame name as reference.
      - Parse its timestamp using FILENAME_RE.
      - Add the time step (inferred from the last two frames, or default 10 minutes)
        successively for each predicted frame.
      - Rebuild new file names that match the original naming convention.

    Parameters
    ----------
    file_list : list of str
        List of input file paths (chronologically sorted).
    n_future : int
        Number of future frames to generate names for.

    Returns
    -------
    list of str
        List of base file names for the predicted frames (without directory path).
    """
    if not file_list:  # If the list is empty we cannot build meaningful names
        # As a last resort, just generate generic names "pred_01.tif", ...
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]  # Simple fallback naming scheme

    last_name = os.path.basename(file_list[-1])  # Get the file name of the last input frame
    match = FILENAME_RE.match(last_name)  # Try to match the naming pattern

    if not match:  # If the pattern does not match, use the simple fallback naming scheme
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]  # No timestamp available, use generic names

    # Extract the timestamp and pieces needed to rebuild the file name
    prefix = match.group("prefix")  # Everything before the date, including the trailing underscore
    suffix = match.group("suffix")  # Everything after the hhmm time (e.g. "_VMI.tiff")
    date_str = match.group("date")  # String in the form YYYYMMDD
    hour_str = match.group("hour")  # Hour as a 2-digit string
    minute_str = match.group("minute")  # Minute as a 2-digit string

    # Build a datetime object for the reference (last input) frame
    base_time = dt.datetime.strptime(date_str + hour_str + minute_str, "%Y%m%d%H%M")  # Combine date and time

    # Infer the temporal step between consecutive frames
    step_minutes = infer_time_step_minutes(file_list)  # Try to deduce the radar sampling interval

    out_names = []  # List that will collect the base names for all predicted frames
    for i in range(1, n_future + 1):  # For each future frame index (1..n_future)
        # Compute the timestamp of the i-th future frame
        future_time = base_time + dt.timedelta(minutes=step_minutes * i)  # Move forward in time

        # Build the new file name using the same style as the original
        future_date = future_time.strftime("%Y%m%d")  # Format as YYYYMMDD
        future_hm = future_time.strftime("%H%M")  # Format as hhmm

        # Reassemble prefix + date + 'Z' + time + suffix
        fname = f"{prefix}{future_date}Z{future_hm}{suffix}"  # Final file name string

        out_names.append(fname)  # Store the generated file name

    return out_names  # Return the list of base names for the predictions


def run_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device, n_future: int) -> torch.Tensor:
    """
    Run the model in inference mode for a single sequence of input frames.

    Parameters
    ----------
    model : torch.nn.Module
        The trained RainPredRNN model.
    input_tensor : torch.Tensor
        Tensor containing the input radar frames. Expected shape is
        (T, C, H, W) or (T, H, W); a batch dimension will be added.
    device : torch.device
        Device where the computation is performed.
    n_future : int
        Number of future frames to predict (n).

    Returns
    -------
    torch.Tensor
        Tensor of predicted frames with shape (n_future, C, H, W).
    """
    # Ensure the input is on the correct device and has a batch dimension
    x = input_tensor.to(device)  # Move input tensor to the same device as the model

    if x.dim() == 3:  # If the input shape is (T, H, W)
        x = x.unsqueeze(1)  # Add a channel dimension -> (T, 1, H, W)
    if x.dim() == 4:  # If the input shape is (T, C, H, W)
        x = x.unsqueeze(0)  # Add a batch dimension -> (1, T, C, H, W)

    # At this point we expect x to be of shape (1, T, C, H, W)
    with torch.no_grad():  # Disable gradient computation for inference
        preds = model(x, n_future=n_future)  # Forward pass through the model

    # Remove the batch dimension to get back to (n_future, C, H, W)
    if preds.dim() == 5 and preds.size(0) == 1:  # If shape is (1, n_future, C, H, W)
        preds = preds.squeeze(0)  # Remove the batch dimension

    return preds  # Return the predicted sequence


def parse_args():
    """
    Define and parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Namespace object containing all parsed arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        description=(
            "RainPred hybrid inference: predict future radar frames from a sequence, "
            "preserving GeoTIFF resolution and naming outputs like input frames."
        )
    )  # Create a new argument parser with a helpful description

    # Path to the trained checkpoint (e.g. best_model.pth)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. best_model.pth).",
    )

    # Directory containing the input radar frames (GeoTIFF files)
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input radar frames (.tif/.tiff).",
    )

    # Number of input frames (m) that the model will use for conditioning
    parser.add_argument(
        "--m",
        type=int,
        default=18,
        help="Number of input frames used for conditioning (m > n).",
    )

    # Number of future frames (n) to predict
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of future frames to predict (n < m).",
    )

    # Output directory where the predicted frames will be stored as GeoTIFFs
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predicted GeoTIFFs will be stored.",
    )

    # Optional flag to force computation on CPU (useful for debugging and systems without CUDA)
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force inference on CPU even if a CUDA GPU is available.",
    )

    args = parser.parse_args()  # Parse the command-line arguments

    return args  # Return the populated namespace


def main():
    """
    Main entry point for command-line execution.

    It parses arguments, loads the model and input data, runs inference,
    and finally saves the predicted radar frames as GeoTIFF files.
    """
    args = parse_args()  # Read all user-specified arguments from the command line

    device = get_device(force_cpu=args.force_cpu)  # Select CPU or GPU based on availability and user preference
    print(f"Using device: {device}")  # Inform the user about the selected device

    # Load the trained model from the checkpoint
    model = load_model(
        checkpoint_path=args.checkpoint,  # Path to the checkpoint file
        device=device,  # Device where the model will be placed
        input_length=args.m,  # Number of input frames used during training/inference
        pred_length=args.n,  # Number of frames the model should predict
    )

    # Load a sequence of input frames from the specified directory.
    # The helper function is expected to:
    #   - read the most recent `m` frames
    #   - return them as a tensor (T, C, H, W) or (T, H, W)
    #   - also return the list of corresponding file paths in chronological order
    input_tensor, file_list = load_sequence_from_dir(
        args.input_dir,  # Directory containing the input GeoTIFF radar frames
        args.m,  # Number of frames to load (m)
    )

    print(f"Loaded {len(file_list)} input frames from: {args.input_dir}")  # Show how many frames were loaded

    # Build human-readable base names for the predicted frames, matching the dataset naming convention
    out_basenames = build_output_basenames(
        file_list=file_list,  # List of input file paths
        n_future=args.n,  # Number of frames to predict
    )

    # Run the neural network inference to obtain the future radar frames
    predictions = run_inference(
        model=model,  # Trained RainPredRNN model
        input_tensor=input_tensor,  # Input radar sequence tensor
        device=device,  # Device used for computation
        n_future=args.n,  # Number of frames to predict
    )

    # Save the predictions as GeoTIFFs, using the same georeferencing as the input frames.
    # The helper is responsible for:
    #   - copying the spatial metadata from the reference files
    #   - mapping prediction indices to file names
    #   - optionally converting the raw output to dBZ if as_dbz=True
    saved_paths = save_predictions_as_geotiff(
        predictions,  # Tensor containing the predicted frames
        reference_file_list=file_list,  # Reference input frames to copy georeferencing from
        output_dir=args.output_dir,  # Destination directory for the output GeoTIFFs
        prefix="pred",  # Prefix used when constructing output names if needed
        as_dbz=True,  # Indicate that predictions are in dBZ or should be converted accordingly
        out_names=out_basenames,  # Explicit base names for the output files (constructed above)
    )

    # Print a short summary of the inference run
    print(f"Checkpoint used: {args.checkpoint}")  # Show which checkpoint was used
    print(f"Predicted {len(saved_paths)} future frames (n={args.n}).")  # Number of frames generated
    print(f"Saved GeoTIFF predictions under: {args.output_dir}")  # Where the files have been stored

    for path in saved_paths:  # For each saved file
        print(f"  -> {os.path.basename(path)}")  # Print only the base name for readability


# Standard Python idiom: execute main() only when the script is run directly,
# not when it is imported as a module.
if __name__ == "__main__":
    main()  # Call the main function when the script is executed from the command line
