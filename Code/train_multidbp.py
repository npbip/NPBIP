import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from typing import List

# Define model parameters
params_dict = {
    "dropout": 0.362233801349954,
    "epochs": 3,          # Number of epochs 72
    "batch": 512,          # Batch size (try 4096, 1024, etc. if needed)
    "regu": 0.0,           # Regularization parameter
    "hidden1": 6029,      #6029
    "hidden2": 1168,      # 1168
    "filters1": 2376,     #2376
    "hidden_sec": 152,    # 152
    "filters_sec": 151,   #151
    "leaky_alpha": 0.23149394545024274,
    "filters_long_length": 24,
    "filters_long": 51
}
params_dict["merge_2"] = params_dict['filters1'] * 4 + params_dict["filters_long"]
params_dict["output_layer"] = (
    params_dict['hidden_sec'] +
    params_dict["hidden2"] +
    params_dict["hidden1"] +
    params_dict["merge_2"]
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiRBPModel(nn.Module):
    """
    A multi-path CNN model for processing one-hot encoded DNA sequences.
    """
    def __init__(self, params_dict, input_dim, output_dim):
        """
        Initialize the MultiRBPModel.
        
        Args:
            params_dict (dict): Dictionary containing model hyperparameters.
            input_dim (int): Number of input channels (e.g. 4 for one-hot encoding).
            output_dim (int): Number of output predictions.
        """
        super(MultiRBPModel, self).__init__()

        # Convolutional layers for the long kernel
        self.conv_kernel_long = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters_long"],
            kernel_size=params_dict["filters_long_length"],
            bias=True,
            padding=0
        )

        # Convolutional layers with different kernel sizes
        self.conv_kernel_11 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters1"],
            kernel_size=11,
            bias=True,
            padding=0
        )
        self.conv_kernel_9 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters1"],
            kernel_size=9,
            bias=True,
            padding=0
        )
        self.conv_kernel_7 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters1"],
            kernel_size=7,
            bias=True,
            padding=0
        )
        self.conv_kernel_5 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters1"],
            kernel_size=5,
            bias=True,
            padding=0
        )
        self.conv_kernel_5_sec = nn.Conv1d(
            in_channels=input_dim,
            out_channels=params_dict["filters_sec"],
            kernel_size=5,
            bias=True,
            padding=0
        )

        # Fully connected layers
        self.hidden_dense_relu = nn.Linear(params_dict["merge_2"], params_dict["hidden1"])
        self.hidden_dense_relu1 = nn.Linear(params_dict["hidden1"], params_dict["hidden2"])
        self.hidden_dense_sec = nn.Linear(params_dict["filters_sec"], params_dict["hidden_sec"])
        self.output_layer = nn.Linear(params_dict["output_layer"], output_dim)

        # Dropout layers
        self.dropout_merge_2 = nn.Dropout(params_dict["dropout"])
        self.dropout_hidden_dense_relu = nn.Dropout(params_dict["dropout"])
        self.dropout_down = nn.Dropout(params_dict["dropout"])
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=params_dict["leaky_alpha"])

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_length, channels)
            
        Returns:
            torch.Tensor: Output predictions.
        """
        # Permute to (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)

        # Convolutional operations
        conv_long = F.relu(self.conv_kernel_long(x))
        conv_11 = F.relu(self.conv_kernel_11(x))
        conv_9 = F.relu(self.conv_kernel_9(x))
        conv_7 = F.relu(self.conv_kernel_7(x))
        conv_5 = F.relu(self.conv_kernel_5(x))
        conv_5_sec = F.relu(self.conv_kernel_5_sec(x))

        # Max pooling operations
        pool_long = F.max_pool1d(conv_long, kernel_size=conv_long.size(-1))
        pool_11 = F.max_pool1d(conv_11, kernel_size=conv_11.size(-1))
        pool_9 = F.max_pool1d(conv_9, kernel_size=conv_9.size(-1))
        pool_7 = F.max_pool1d(conv_7, kernel_size=conv_7.size(-1))
        pool_5 = F.max_pool1d(conv_5, kernel_size=conv_5.size(-1))
        pool_5_sec = F.max_pool1d(conv_5_sec, kernel_size=conv_5_sec.size(-1))
        

        # First path: merge multiple pooling outputs
        merge_first_path = torch.cat([pool_11, pool_7, pool_long, pool_9, pool_5], dim=1)
        merge_first_path_flatten = merge_first_path.squeeze()
        merge_first_path_flatten_dropout = self.dropout_merge_2(merge_first_path_flatten)
        merge_dense_a = F.relu(self.hidden_dense_relu(merge_first_path_flatten_dropout))
        merge_dense_a_dropout = self.dropout_hidden_dense_relu(merge_dense_a)
        merge_dense_b = F.relu(self.hidden_dense_relu1(merge_dense_a_dropout))

        # Second path: process secondary convolutional path
        second_path_flatten = pool_5_sec.squeeze()
        second_path_dropout = self.dropout_down(second_path_flatten)
        second_dense = F.relu(self.hidden_dense_sec(second_path_dropout))

        # Final merge of paths
        merge_final = torch.cat([second_dense, merge_dense_b, merge_first_path_flatten_dropout, merge_dense_a], dim=1)

        # Output layer with activation
        y = self.output_layer(merge_final)
        y = self.leaky_relu(y)
        return y


class DNAProbes(Dataset):
    """
    Dataset for DNA probes. Uses a dictionary mapping DNA sequences to their one-hot encodings
    and a DataFrame with corresponding target values.
    """
    def __init__(self, dna_to_one_hot, full_data):
        """
        Args:
            dna_to_one_hot (dict): Dictionary where keys are DNA sequences and values are one-hot encodings.
            full_data (pd.DataFrame): DataFrame with columns 'DNA_Seq' and target columns.
        """
        if 'DNA_Seq' not in full_data.columns:
            raise KeyError("Column 'DNA_Seq' not found in the provided DataFrame.")
        self.dna_seq = full_data['DNA_Seq'].tolist()
        self.dna_to_one_hot = dna_to_one_hot
        # Drop the DNA sequence column from targets
        self.full_data = full_data.drop('DNA_Seq', axis=1)
        
    def __len__(self):
        return len(self.dna_seq)

    def __getitem__(self, idx):
        """
        Retrieve the one-hot encoded DNA probe and its corresponding target values.
        """
        dna_probe = self.dna_seq[idx]
        one_hot = self.dna_to_one_hot[dna_probe]
        y = torch.tensor(self.full_data.iloc[idx].values, dtype=torch.float32)
        return one_hot, y


def train(dataloader, model, loss_fn, optimizer, p_in_test, test_loader):
    """
    Train the model for one epoch.
    
    Args:
        dataloader (DataLoader): Training data loader.
        model (nn.Module): The model to train.
        loss_fn: Loss function.
        optimizer: Optimizer.
        p_in_test (int): Output dimension expected.
    
    Returns:
        float: Total loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (dna, y) in enumerate(dataloader):
        dna, y = dna.to(device), y.to(device)
        y = y.reshape(-1, p_in_test)

        # Forward pass and loss computation
        pred = model(dna)
        loss = loss_fn(pred, y)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss


def test(dataloader, model, p_in_test):
    """
    Evaluate the model on the provided dataloader.
    
    Args:
        dataloader (DataLoader): Data loader for evaluation.
        model (nn.Module): The model to evaluate.
        p_in_test (int): Expected output dimension.
    
    Returns:
        tuple: (predictions, targets, mean Pearson correlation)
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for dna, y in dataloader:
            dna, y = dna.to(device), y.to(device)
            y = y.reshape(-1, p_in_test)
            pred = model(dna)
            preds.append(pred)
            targets.append(y)
    
    preds = torch.cat(preds, dim=0).squeeze().detach().cpu().numpy()
    targets = torch.cat(targets, dim=0).squeeze().detach().cpu().numpy()

    # Compute Pearson correlation for each output dimension and average
    correlations = []
    for i in range(p_in_test):
        score = stats.pearsonr(targets[:, i], preds[:, i])[0]
        correlations.append(score)
    mean_pearson = np.mean(correlations)
    print(f"Mean Pearson correlation: {mean_pearson:.4f}")
    
    return preds, targets, mean_pearson


class LogCoshLoss(nn.Module):
    """
    Custom Log-Cosh loss.
    """
    def __init__(self, reduction='mean'):
        """
        Initialize LogCoshLoss.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output: 'mean', 'sum', or 'none'.
        """
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        # Clamp to prevent overflow in cosh
        diff = torch.clamp(diff, min=-80, max=80)
        loss = torch.log(torch.cosh(diff))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


# Define the nucleic acids order and mapping (adjust if needed)
NUCLEIC_ACIDS = 'ACGT'
NN_TO_IX = {nucleotide: i for i, nucleotide in enumerate(NUCLEIC_ACIDS)}


def seq_to_indices(sequence: str) -> List[int]:
    """
    Convert a DNA/RNA sequence into a list of indices based on NUCLEIC_ACIDS order.

    Parameters:
        sequence (str): The input DNA/RNA sequence.

    Returns:
        List[int]: A list of indices corresponding to the sequence.

    Raises:
        ValueError: If the sequence contains an unknown nucleotide.
    """
    indices = []
    for nucleotide in sequence:
        if nucleotide not in NN_TO_IX:
            raise ValueError(f"Unexpected nucleotide '{nucleotide}' in sequence '{sequence}'.")
        indices.append(NN_TO_IX[nucleotide])
    return indices


def one_hot_encode(
    indices: List[int],
    padding: bool = False,
    seq_length: int = 0,
    padding_type: str = 'equal'
) -> np.ndarray:
    """
    Create a one-hot encoded tensor from a list of indices.

    Parameters:
        indices (List[int]): List of indices representing the sequence.
        padding (bool): Whether to pad the tensor to seq_length.
        seq_length (int): The length of the tensor after padding.
        padding_type (str): 'equal' to initialize padded rows uniformly (1/4 each),
                            otherwise (e.g., 'zero') to initialize padded rows with zeros.

    Returns:
        np.ndarray: A one-hot encoded tensor of shape (len(indices), 4) if padding is False,
                    or (seq_length, 4) if padding is True.
    """
    if not padding:
        tensor = np.zeros((len(indices), 4), dtype=np.float32)
    elif padding_type == 'equal':
        tensor = np.ones((seq_length, 4), dtype=np.float32) / 4
    else:
        tensor = np.zeros((seq_length, 4), dtype=np.float32)

    # Encode each index into the tensor (ensuring we don't exceed seq_length when padded)
    for i, idx in enumerate(indices):
        if padding and i >= seq_length:
            break
        tensor[i] = 0  # Reset the row
        tensor[i, idx] = 1  # Set the one-hot encoding
    return tensor


def encode_dna(
    rna: str,
    padding: bool = False,
    seq_length: int = 0,
    padding_type: str = 'zero'
) -> np.ndarray:
    """
    Encode a DNA sequence as a one-hot encoded tensor.

    Parameters:
        rna (str): The DNA/RNA sequence.
        padding (bool): Whether to pad the tensor.
        seq_length (int): The desired sequence length after padding.
        padding_type (str): Padding initialization method ('equal' or other).

    Returns:
        np.ndarray: A one-hot encoded tensor.
    """
    indices = seq_to_indices(rna)
    return one_hot_encode(indices, padding, seq_length, padding_type)


def find_available_filename(base_path: str) -> str:
    """
    Checks if a file exists and appends _1, _2, etc., until a non-existing filename is found.

    Parameters:
        base_path (str): The original file path (e.g., 'output.txt').

    Returns:
        str: A modified file path that does not exist.
    """
    if not os.path.exists(base_path):
        return base_path

    name, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{name}_{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def main():

    parser = argparse.ArgumentParser(description="Train multidbp")
    parser.add_argument("data", help="csv file of PBM data - first columns dna sequences named 'DNA_Seq' other columns scores")
    args = parser.parse_args()

    data_file = args.data
    full_data = pd.read_csv(data_file)

    probes = full_data["DNA_Seq"].to_list()
    dna_probe_to_one_hot = {probe:encode_dna(probe, padding=True, seq_length=41, padding_type='zero') 
                            for i, probe in enumerate(probes)}

    # Determine output dimension (all columns except DNA_Seq)
    p_in_test = full_data.shape[1] - 1
    test_data = full_data.iloc[-10000:]
    train_data = full_data.iloc[:-10000]
    
    # Create Dataset and DataLoader objects
    train_dataset = DNAProbes(dna_probe_to_one_hot, train_data)
    test_dataset = DNAProbes(dna_probe_to_one_hot, test_data)
    train_loader = DataLoader(train_dataset, batch_size=params_dict['batch'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MultiRBPModel(params_dict, input_dim=4, output_dim=p_in_test).to(device)
    print(f"Model total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    loss_fn = LogCoshLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(params_dict['epochs']):
        print(f"\nEpoch {epoch + 1}/{params_dict['epochs']}")
        train_loss = train(train_loader, model, loss_fn, optimizer, p_in_test, test_loader)


    print("finsh train!")
    # Evaluate on training and test sets
    print("\nEvaluating on training data:")
    _, _, train_corr = test(train_loader, model, p_in_test)
    
    print("\nEvaluating on test data:")
    preds, targets, test_corr = test(test_loader, model, p_in_test)
    

    # Save the model
    model_filename = f"multidbp.pt"
    model_filename = find_available_filename(model_filename)
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'.")
    

if __name__ == "__main__":
    main()