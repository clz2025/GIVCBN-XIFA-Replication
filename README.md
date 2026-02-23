# XIFA: X-Index Filtering Algorithm Implementation

This module implements an X-Index-based network pruning algorithm (X-Index Filtering Algorithm, XIFA) for extracting backbone subnetworks from weighted directed networks. The algorithm preserves the most important outgoing edges for each node, removing redundant connections to reveal the core structure of the network.

## Algorithm Overview

The core idea of XIFA originates from the concept of the "X-Index" in complex networks. For a weighted directed network, each node calculates a threshold index `A[i]` based on the descending order of its outgoing edge weights, satisfying specific conditions. Only the top `A[i]` most important outgoing edges of the node are retained; the rest are set to zero.

The X-Index is defined as follows:
> For node `i`, sort its outgoing edge weights in descending order to obtain vector `VEC`. Let the total weight be `W = sum(VEC)`. Iterate starting from `a = 1` to find the smallest `a` that satisfies the condition:
> `sum(VEC[:a]) / W >= 1 - a/N`
> (where `N` is the total out-degree of the node). Then, `XI[i] = 100 * a / N`, and the threshold index `A[i] = a`.

This condition ensures that the retained edges contribute a sufficient proportion of the total weight while preventing the network from becoming overly dense.

The algorithm consists of three main steps:
1.  Calculate the X-Index and the corresponding threshold index `A` for each node.
2.  For each node, retain only the top `A` edges by weight and set the rest to zero.
3.  Output the pruned network matrix.

This algorithm is parameter-free and entirely data-driven, suitable for extracting the backbone structure of weighted networks.

## File Description

`PrunNet_FilteringEdges_Combo(1).py` contains the following functions:

- **`PrunNet_FilteringEdges(input_file_path, output_folder_path)`**  
  The main function. Reads the input file, applies XIFA to both rows and columns (processing the original matrix and its transpose separately), and takes the intersection (union of non-zero elements) of the two results as the final filtered network. Outputs the result as a CSV file and prints statistical information.

- **`PrunNet_from_XI(net)`**  
  A subroutine. Takes a matrix `net`, calculates the X-Index row by row, performs pruning, and returns the pruned matrix, the number of retained edges, and the retained weight ratio.

- **`X_Index(net)`**  
  A subroutine. Calculates the X-Index value `XI` and the threshold index `A` for each row of the input matrix `net`.

## Dependencies

- `numpy`
- `pandas`
- `tqdm`
- `time`
- `os`

## Input Requirements

- Input file formats supported: `.csv`, `.xlsx`, `.xls`.
- The file should be a square or rectangular matrix, with the first row and first column containing node names (i.e., using `index_col=0` in Pandas).
- Matrix elements must be numerical values representing edge weights (e.g., intermediate goods trade flows). Missing values or negative numbers are converted to 0 during processing.

## Output Description

- The output file is saved in the specified output folder with the filename format: `original_filename-FE.csv`.
- The output matrix has the same dimensions as the input matrix. Only edges satisfying the XIFA condition are retained; all other entries are set to 0.
- The console prints the following statistics (based on actual runtime calculations):
  - File reading time
  - Total computation time
  - File saving time
  - Number of edges in the original network (number of non-zero elements)
  - Number of edges after filtering
  - Retained weight ratio (total weight after filtering / total original weight × 100%)

## Usage

### Running the Script Directly

Modify the `input_file` and `output_folder` variables at the end of the script, then run:

```python
if __name__ == '__main__':
    input_file = r"path_to_your_input_file"
    output_folder = r"path_to_your_output_folder"
    FE = PrunNet_FilteringEdges(input_file, output_folder)
```

### Calling as a Module

```python
from PrunNet_FilteringEdges_Combo import PrunNet_FilteringEdges

result_df = PrunNet_FilteringEdges("data.csv", "./output")
```

## Algorithm Details

### X-Index Calculation

The X-Index is calculated by iterating through each possible `a` (from 1 to N) to find the first `a` satisfying the inequality. If a row consists entirely of zeros, then `XI[i]=0` and `A[i]=0`.

### Pruning Process

For the i-th row, the weights are sorted in descending order to obtain the index array `Xidx`. A reverse mapping `Xidx2` is created so that `Xidx2[j]` indicates the sorted position of the original column `j`. If this position is less than `A[i]`, the original weight `net[i, j]` is retained; otherwise, it is set to 0.

### Row and Column Combination

The main function applies pruning to the original matrix and its transpose separately, yielding two pruned matrices, `PrunNet1` and `PrunNet2`. These two matrices are summed (`PrunNet1 + PrunNet2.T`). The positions of non-zero elements in the sum indicate the edges retained in the final network (the union). Finally, an element-wise multiplication with the original matrix restores the original weights for the retained edges.

## Notes

- The algorithm assumes the input matrix represents a weighted directed network, where larger weights indicate stronger relationships.
- The pruning process preserves the local importance of each node; therefore, the final network remains directed.
- For large-scale matrices (e.g., thousands of nodes), computation time may be significant. A high-performance computing environment is recommended.
- Progress bars (`tqdm`) in the code display the processing status in real-time.


# GIVCBN-XIFA-Replication
