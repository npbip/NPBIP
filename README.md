# 🔬 NPBIP: Predicting Protein–Nucleic Acid Binding Interactions

NPBIP is a tool for predicting binding interactions between proteins and nucleic acids (DNA or RNA).

---

## 📦 Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

In addition, the program **requires** the external tool [`hmmscan`](http://hmmer.org/), part of the [HMMER](http://hmmer.org/) suite.

### 🔧 Installing `hmmscan`

#### 🐧 On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install hmmer
```

#### 🍎 On macOS with Homebrew:

```bash
brew install hmmer
```

#### 🖐️ Manual Installation:

Download and install from [http://hmmer.org/download.html](http://hmmer.org/download.html):

```bash
./configure
make
sudo make install
```

Verify installation:

```bash
hmmscan -h
```

---

## 🚀 Usage

To run the main prediction script:

```bash
python Code/main_NPBIP.py <nucleic_acid_type> <protein_fasta> <nucleic_acid_fasta> [--run_id RUN_ID]

```

### 🧾 Parameters:

- `<nucleic_acid_type>`: Either `DNA` or `RNA`
- `<protein_fasta>`: Path to a FASTA file containing **query protein sequences**
- `<nucleic_acid_fasta>`: Path to a FASTA file containing **nucleic acid sequences** (DNA or RNA probes)
- `--run_id` (optional): Custom name for this run. All results will be saved under output/<run_id>/. If not provided, a default name will be generated.

### 💡 Example:

```bash
python Code/main_NPBIP.py DNA examples/DBPs.fasta examples/dna.fasta
python Code/main_NPBIP.py RNA examples/RBPs.fasta examples/rna.fasta --run_id run1_rna
```


### 🧩 Pipeline Overview
The `main_NPBIP.py` script orchestrates the full prediction pipeline by calling four main modules:

1. `protein_domain.py`  
   Detects nucleic acid binding domains in the query proteins.

2. `pair_wise.py`  
   Computes similarity scores between the query proteins and training proteins.

3. `NewSeq_prediction.py`  
   Predicts binding intensity for the query nucleic acid sequences with training protein.

4. `NewProNewSeq_final_prediction.py`  
   Computes a weighted average of the predictions based on the similarity scores to generate the final output.


---

## ✨ License

This project is open-source and free to use under the MIT License.
