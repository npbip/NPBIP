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
python main_NPBIP.py <nucleic_acid_type> <protein_fasta> <nucleic_acid_fasta>
```

### 🧾 Parameters:

- `<nucleic_acid_type>`: Either `DNA` or `RNA`
- `<protein_fasta>`: Path to a FASTA file containing **query protein sequences**
- `<nucleic_acid_fasta>`: Path to a FASTA file containing **nucleic acid sequences** (DNA or RNA probes)

### 💡 Example:

```bash
python main_NPBIP.py RNA examples/proteins.fasta examples/rna_sequences.fasta
```

---

## ✨ License

This project is open-source and free to use under the MIT License.
