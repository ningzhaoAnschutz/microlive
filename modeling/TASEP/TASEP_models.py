import sys; from pathlib import Path
from Bio.SeqRecord import SeqRecord
from collections import Counter
from Bio.Data import CodonTable
import json
src_dir = next((parent / 'src' for parent in Path().absolute().parents if (parent / 'src').is_dir()), None)
sys.path.append(str(src_dir))
from imports import *


os.environ["OPENBLAS_NUM_THREADS"] = "1"
human_genome_path = src_dir.parents[0].joinpath('modeling/TASEP/human_genome/Homo_sapiens.GRCh38.cds.all.fa')

def download_human_genome_cds (human_genome_path):
    if human_genome_path.exists() == False:
        print("human_genome_path does not exist. Downloading...")
        human_genome_dir = human_genome_path.parent
        human_genome_dir.mkdir(parents=True, exist_ok=True)
        url = ("ftp://ftp.ensembl.org/pub/release-108/fasta/"
            "homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz" )
        gz_path = human_genome_dir / (human_genome_path.name + ".gz")
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(human_genome_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()



GFP_TAG = 'AAGITH' #'TYA'
HA_TAG = 'YPYDVPDYA'
U_TAG = 'MSLPGRWKPKM'
SUN_TAG = 'EELLSKNYHLENEVARLKK'
ALFA_TAG = 'SRLEEELRRRLTE'
MCHERRY_TAG = 'EGRHSTG'
XBP1 = 'KDPVPYQPPFLCQWGRHQPAWKPLMN'


# create a dictionary with the tag sequences.
tag_dict = {'GFP': GFP_TAG, 'HA': HA_TAG, 'U': U_TAG, 'SUN': SUN_TAG, 'ALFA': ALFA_TAG, 'mCherry': MCHERRY_TAG}

# Sequences that pause ribosome elongation
pause_dict = {'XBP1': XBP1}

# Loading human codon weights as global
# TASEP_DIR = src_dir.parents[0].joinpath('modeling/TASEP')
# CODON_WEIGHTS_PATH = TASEP_DIR / 'human_codon_weights.json'

# determine where this file is located
TASEP_DIR = Path(__file__).resolve().parent
CODON_WEIGHTS_PATH = TASEP_DIR / 'human_codon_weights.json'


try:
    with open(CODON_WEIGHTS_PATH, 'r') as f:
        HUMAN_CODON_WEIGHTS = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find 'human_codon_weights.json' at {CODON_WEIGHTS_PATH}. "
        f"Please ensure the file exists in the TASEP directory: {TASEP_DIR}"
    )

def simulate_missing_data(matrix1, matrix2=None, percentage_to_remove_data=0, replace_with='nan'):
    if percentage_to_remove_data ==0: 
        return matrix1, matrix2
    if matrix2 is not None:
        if matrix1.shape != matrix2.shape:
            raise ValueError("Both matrices must have the same shape.")
    num_rows, num_cols = matrix1.shape
    new_matrix1 = matrix1.copy()
    if matrix2 is not None:
        new_matrix2 = matrix2.copy()
    #if total_cols_to_remove >= num_cols:
    #    raise ValueError("Percentage to remove too high, no columns left to keep.")
    # Determine replacement value (zero or NaN)
    if replace_with == 'zeros':
        replacement_value = 0
    elif replace_with == 'nan':
        replacement_value = np.nan
    else:
        raise ValueError("Invalid replace_with argument. Use 'zeros' or 'nan'.")
    for i in range(num_rows):
        #total_cols_to_remove = int(num_cols * (percentage_to_remove_data / 100))
        # Randomly select columns to remove between 20% of the percentage_to_remove_data
        rand_percentage_to_remove_data = np.random.randint(int(0.5*percentage_to_remove_data), int(1.5*percentage_to_remove_data))
        total_cols_to_remove = int(num_cols * (rand_percentage_to_remove_data / 100))
        total_cols_to_remove = min(total_cols_to_remove, num_cols)  # Ensure not removing more columns than available
        # Randomly split the total columns to remove between left and right
        left_cols_to_remove = np.random.randint(0, total_cols_to_remove + 1)
        right_cols_to_remove = total_cols_to_remove - left_cols_to_remove
        # Replace the columns from the extremes in both matrices
        if left_cols_to_remove > 0:
            new_matrix1[i, :left_cols_to_remove] = replacement_value
            if matrix2 is not None:
                new_matrix2[i, :left_cols_to_remove] = replacement_value
        if right_cols_to_remove > 0:
            new_matrix1[i, num_cols - right_cols_to_remove:] = replacement_value
            if matrix2 is not None:
                new_matrix2[i, num_cols - right_cols_to_remove:] = replacement_value
    if matrix2 is None:
        return new_matrix1, None # Return only the first matrix if the second one is None
    else:   
        return new_matrix1, new_matrix2
    
def simulate_photobleaching_in_trajectories(matrix, decay_rate):
    num_rows, num_cols = matrix.shape
    # Generate the time points (column indices) for the decay
    time_points = np.arange(num_cols)
    # Calculate the exponential decay factor for each time point
    decay_factors = np.exp(-decay_rate * time_points)
    # Apply the decay equally to each row
    decayed_matrix = matrix * decay_factors
    return decayed_matrix

# correct for the photobleaching
def correct_photobleaching_in_trajectories(matrix, decay_rate):
    num_rows, num_cols = matrix.shape
    # Generate the time points (column indices) for the decay
    time_points = np.arange(num_cols)
    # Calculate the exponential decay factor for each time point
    decay_factors = np.exp(-decay_rate * time_points)
    # Apply the decay equally to each row
    corrected_matrix = matrix / decay_factors
    return corrected_matrix

def delay_signal(signal, time_delay):
    # Create a delay as an array of zeros
    delay = np.zeros(time_delay)
    # Concatenate delay to the beginning of the signal
    delayed_signal = np.concatenate((delay, signal))
    # Remvoing the end of the signal.
    delayed_signal = delayed_signal[:len(signal)]
    return delayed_signal

# Codon usage data from: https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606
human_codon_frequency = """
UUU 17.6  UCU 15.2  UAU 12.2  UGU 10.6
UUC 20.3  UCC 17.7  UAC 15.3  UGC 12.6
UUA  7.7  UCA 12.2  UAA  1.0  UGA  1.6
UUG 12.9  UCG  4.4  UAG  0.8  UGG 13.2
CUU 13.2  CCU 17.5  CAU 10.9  CGU  4.5
CUC 19.6  CCC 19.8  CAC 15.1  CGC 10.4
CUA  7.2  CCA 16.9  CAA 12.3  CGA  6.2
CUG 39.6  CCG  6.9  CAG 34.2  CGG 11.4
AUU 16.0  ACU 13.1  AAU 17.0  AGU 12.1
AUC 20.8  ACC 18.9  AAC 19.1  AGC 19.5
AUA  7.5  ACA 15.1  AAA 24.4  AGA 12.2
AUG 22.0  ACG  6.1  AAG 31.9  AGG 12.0
GUU 11.0  GCU 18.4  GAU 21.8  GGU 10.8
GUC 14.5  GCC 27.7  GAC 25.1  GGC 22.2
GUA  7.1  GCA 15.8  GAA 29.0  GGA 16.5
GUG 28.1  GCG  7.4  GAG 39.6  GGG 16.5 
"""

codon_frequency_dict = {}
for line in human_codon_frequency.strip().split('\n'):
    parts = line.split()
    for i in range(0, len(parts), 2):
        codon = parts[i] 
        frequency = float(parts[i + 1])  
        codon_frequency_dict[codon] = frequency  

# dictionary of synonymous codons mixed T/U
synonymous_codons = {
                'A':['GCA', 'GCC', 'GCG', 'GCT', 
                                          'GCU'],
                'R':['CGT', 'CGA', 'CGC', 'CGG',  'AGG', 'AGA', 
                     'CGU'],
                'N':['AAC', 'AAT',
                            'AAU'],
                'D':['GAC', 'GAT', 
                            'GAU'],
                'C':['TGC', 'TGT', 
                     'UGC', 'UGU'],
                'Q':['CAA', 'CAG'],
                'E':['GAA', 'GAG'],
                'G':['GGT', 'GGC', 'GGA', 'GGG', 
                     'GGU'],
                'H':['CAC', 'CAT',
                            'CAU'],
                'I':['ATT', 'ATC', 'ATA', 
                     'AUU', 'AUC', 'AUA'],
                'L':['CTA', 'CTC', 'CTG', 'CTT', 'TTA', 'TTG', 
                     'CUA', 'CUC', 'CUG', 'CUU', 'UUA', 'UUG'],
                'K':['AAA', 'AAG'],
                'M':['ATG', 
                     'AUG'],
                'F':['TTC', 'TTT',
                     'UUC', 'UUU'],
                'P':['CCT', 'CCC', 'CCG', 'CCA', 'CCU'],
                'S':['TCA', 'TCC', 'TCG', 'TCT', 'AGT', 'AGC',
                     'UCA', 'UCC', 'UCG', 'UCU', 'AGU'],
                'T':['ACA', 'ACC', 'ACG', 'ACT',
                                          'ACU'],
                'W':['TGG', 
                     'UGG'],
                'Y':['TAT', 'TAC', 
                     'UAU', 'UAC'],
                'V':['GTA', 'GTC', 'GTT', 'GTG', 
                     'GUA', 'GUG', 'GUU', 'GUC'],
                '*':['TGA', 'TAG', 'TAA', 
                     'UGA', 'UAG', 'UAA']
                }

# 1) First, normalize your usage table to U-based codons (since your Kazusa data uses U)
#    and build your frequency dict as you already have.

# 2) Build a codon→amino-acid map by inverting your synonymous_codons dict:
codon_to_aa = {}
for aa, codons in synonymous_codons.items():
    for c in codons:
        # normalize to RNA form for lookup
        rna_c = c.replace('T', 'U')
        codon_to_aa[rna_c] = aa


def deoptimize_sequence(sequence):
    """
    Deoptimizes an RNA sequence by replacing each codon with the
    least-frequent synonymous codon.
    """
    download_human_genome_cds (human_genome_path)
    seq = sequence.upper().replace('T', 'U')  # work in RNA
    out = []
    for i in range(0, len(seq), 3):
        cod = seq[i:i+3]
        aa = codon_to_aa.get(cod)
        if aa:
            # get synonyms (also normalized to RNA) that we actually have frequencies for
            syns = [c.replace('T','U') for c in synonymous_codons[aa]
                    if c.replace('T','U') in codon_frequency_dict]
            if syns:
                # pick the least-frequent one
                min_codon = min(syns, key=lambda x: codon_frequency_dict[x])
                out.append(min_codon)
                continue
        # fallback: either non-standard codon or no synonyms → leave as-is
        out.append(cod)
    # if you want DNA output, convert U→T here:
    return ''.join(out).replace('U', 'T')

def optimize_sequence(sequence, ):
    """
    Optimizes an RNA (or DNA) coding sequence by replacing each codon
    with the highest-frequency synonymous codon.

    Args:
        sequence (str): Input DNA/RNA sequence.
        codon_frequency_dict (dict): Mapping RNA codon -> frequency.
        synonymous_codons (dict): AA -> list of codons (T/U mixed).

    Returns:
        str: Optimized sequence in DNA alphabet (T).
    """
    # normalize to RNA
    download_human_genome_cds (human_genome_path)
    seq_rna = sequence.upper().replace('T', 'U')
    out = []
    for i in range(0, len(seq_rna), 3):
        cod = seq_rna[i:i+3]
        aa = codon_to_aa.get(cod)
        if aa:
            # gather synonyms in RNA form that we have frequencies for
            syns = [
                syn.replace('T','U')
                for syn in synonymous_codons[aa]
                if syn.replace('T','U') in codon_frequency_dict
            ]
            if syns:
                # pick the most-used one
                best = max(syns, key=lambda x: codon_frequency_dict[x])
                out.append(best)
                continue
        # fallback: emit original codon
        out.append(cod)
    # convert back to DNA for output
    return ''.join(out).replace('U', 'T')



def compute_CAI(sequence):
    """
    Matches BioPython's calculation method.
    Exclude Met (ATG) and Trp (TGG) which have only one codon.
    """
    sequence = sequence.upper().replace("U", "T")
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    # Codons to exclude (following Sharp & Li 1987 convention)
    STOP_CODONS = {'TAA', 'TAG', 'TGA'}
    SINGLE_CODON_AA = {'ATG', 'TGG'}  # Met and Trp - only one codon each
    valid_weights = []
    for i, codon in enumerate(codons):
        # Skip first codon (start)
        if i == 0:
            continue
        if codon in STOP_CODONS:
            continue
        if codon in SINGLE_CODON_AA:
            continue
        if codon in HUMAN_CODON_WEIGHTS and HUMAN_CODON_WEIGHTS[codon] > 0:
            valid_weights.append(HUMAN_CODON_WEIGHTS[codon])
    if not valid_weights:
        return 0.0
    # Geometric mean
    return math.exp(sum(math.log(w) for w in valid_weights) / len(valid_weights))


def sliding_window_cai(sequence, window_size=30, step=1):
    """
    Calculate CAI for sliding windows along a sequence.
    
    Parameters:
    -----------
    sequence : str
        DNA sequence (must be divisible by 3)
    window_size : int
        Window size in codons (default: 30)
    step : int
        Step size in codons (default: 1)
    
    Returns:
    --------
    positions : np.ndarray
        Center position of each window (in codons)
    cai_values : np.ndarray
        CAI value for each window
    """
    # Ensure sequence length is divisible by 3
    seq_length_codons = len(sequence) // 3
    positions = []
    cai_values = []
    # Slide the window
    for start_codon in range(0, seq_length_codons - window_size + 1, step):
        # Extract window in nucleotides (3 bases per codon)
        start_nt = start_codon * 3
        end_nt = (start_codon + window_size) * 3
        window_seq = sequence[start_nt:end_nt]
        try:
            cai = compute_CAI(window_seq)
            positions.append(start_codon + window_size / 2)  # center of window
            cai_values.append(cai)
        except:
            continue
    return np.array(positions), np.array(cai_values)


def plot_sliding_window_cai(
    sequences, 
    sequence_names=None, 
    window_size=20, 
    step=1,
    figsize=(12, 6),
    save_path=None,
    color_map = None,
    markers = None

):
    """
    Plot sliding window CAI for multiple sequences on the same plot.
    
    Parameters:
    -----------
    sequences : list of str
        List of DNA sequences
    sequence_names : list of str
        Names for each sequence
    window_size : int
        Window size in codons (default: 30 codons = 90 bp)
    step : int
        Step size in codons (default: 1)
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the figure
    """
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    if not isinstance(sequences, list):
        sequences = [sequences] 
    if color_map is None:
        color_map = 'tab10'
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i) for i in range(len(sequences))]

    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    if sequence_names is None:
        sequence_names = [f'Sequence {i+1}' for i in range(len(sequences))]

    # Plot each sequence
    for idx, (seq, name) in enumerate(zip(sequences, sequence_names)):
        positions, cai_values = sliding_window_cai(
            seq, window_size=window_size, step=step
        )
        marker = markers[idx % len(markers)]
        ax.plot(positions, cai_values, 
                marker=marker,           # Add marker
                linestyle='-',           # Line style
                color=colors[idx], 
                linewidth=3, 
                markersize=6,            # Marker size
                markevery=5,             # Show marker every N points (adjust for clarity)
                label=name, 
                alpha=0.8)
    # Formatting
    ax.set_xlabel(f'Position (codons)', fontsize=18)
    ax.set_ylabel(f'CAI (window = {window_size} codons)', fontsize=18)
    ax.set_title(f'Sliding Window CAI Comparison', fontsize=18)
    ax.legend(loc='upper right', fontsize=14, frameon=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    # y-axis limits
    ax.set_ylim(0.2, 1.2)
    # Set spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax




def calculate_codon_usage(sequence):
    """
    Calculate codon usage frequency for a DNA sequence.
    
    Parameters:
    -----------
    sequence : str
        DNA sequence (must be divisible by 3)
    
    Returns:
    --------
    codon_counts : dict
        Dictionary with codon counts
    codon_frequencies : dict
        Dictionary with codon frequencies (normalized by amino acid)
    """
    # Extract codons
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    
    # Count codons
    codon_counts = Counter(codons)
    
    # Get standard genetic code
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
    
    # Group codons by amino acid
    aa_to_codons = {}
    for codon, aa in standard_table.forward_table.items():
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        aa_to_codons[aa].append(codon)
    
    # Add stop codons
    aa_to_codons['*'] = ['TAA', 'TAG', 'TGA']
    
    # Calculate frequencies per amino acid
    codon_frequencies = {}
    for aa, codon_list in aa_to_codons.items():
        total_aa_codons = sum(codon_counts.get(codon, 0) for codon in codon_list)
        for codon in codon_list:
            if total_aa_codons > 0:
                codon_frequencies[codon] = codon_counts.get(codon, 0) / total_aa_codons
            else:
                codon_frequencies[codon] = 0
    
    return codon_counts, codon_frequencies


def plot_codon_usage_grouped(
    sequences, 
    sequence_names, 
    x_label_type='aa_codon',
    figsize=(28, 8),
    save_path=None,
    include_stop_codons=True,
    color_map='tab10'
):
    """
    Plot codon usage frequency for multiple sequences as grouped bars.
    
    Parameters:
    -----------
    sequences : list of str
        List of DNA sequences
    sequence_names : list of str
        Names for each sequence
    x_label_type : str, optional
        Type of x-axis labels:
        - 'codon': Show 3-letter codon names (e.g., 'ATG')
        - 'aa': Show 1-letter amino acid codes only (e.g., 'M')
        - 'aa_codon': Show amino acid with codon (e.g., 'M(ATG)')
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save the figure
    include_stop_codons : bool, optional
        Whether to include stop codons in the plot (default: True)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    if x_label_type not in ['codon', 'aa', 'aa_codon']:
        raise ValueError("x_label_type must be 'codon', 'aa', or 'aa_codon'")
    
    # Get standard genetic code
    standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
    
    # Create amino acid to codon mapping
    aa_to_codons = {}
    for codon, aa in standard_table.forward_table.items():
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        aa_to_codons[aa].append(codon)
    
    # Add start codon (Methionine)
    aa_to_codons['M'] = ['ATG']
    
    if include_stop_codons:
        aa_to_codons['*'] = ['TAA', 'TAG', 'TGA']
    
    # Sort amino acids (M first, then alphabetically, then stop codon last)
    amino_acids = ['M'] + sorted([aa for aa in aa_to_codons.keys() if aa not in ['M', '*']])
    if include_stop_codons:
        amino_acids.append('*')  # Add stop codon at the end
    
    # Calculate codon usage for all sequences
    all_frequencies = []
    for seq in sequences:
        _, codon_freq = calculate_codon_usage(seq)
        all_frequencies.append(codon_freq)
    
    # Prepare data and labels
    all_codons = []
    x_labels = []
    
    for aa in amino_acids:
        codons_for_aa = sorted(aa_to_codons[aa])
        all_codons.extend(codons_for_aa)
        
        # Build labels based on user choice
        if x_label_type == 'codon':
            x_labels.extend(codons_for_aa)
        elif x_label_type == 'aa':
            x_labels.extend([aa] * len(codons_for_aa))
        elif x_label_type == 'aa_codon':
            # Use "STOP" instead of "*" for better readability
            aa_display = 'STOP' if aa == '*' else aa
            x_labels.extend([f"{aa_display}({codon})" for codon in codons_for_aa])
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    n_sequences = len(sequences)
    bar_width = 0.8 / n_sequences
    x = np.arange(len(all_codons))
    
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i) for i in range(n_sequences)] 
    
    for idx, (freq_dict, name) in enumerate(zip(all_frequencies, sequence_names)):
        frequencies = [freq_dict.get(codon, 0) for codon in all_codons]
        offset = (idx - n_sequences/2) * bar_width + bar_width/2
        ax.bar(x + offset, frequencies, bar_width, 
               label=name, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Formatting
    xlabel_dict = {
        'codon': 'Codon',
        'aa': 'Amino Acid',
        'aa_codon': 'Amino Acid (Codon)'
    }
    ax.set_xlabel(xlabel_dict[x_label_type], fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    #ax.set_title('Codon Usage Comparison', fontsize=20, fontweight='bold', loc='left')
    ax.set_xticks(x)
    
    # Adjust font size based on label type
    label_fontsize = 14
    ax.set_xticklabels(x_labels, rotation=45, fontsize=label_fontsize, ha='right')
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-1, len(all_codons) +2)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax




def find_TAG_location(protein, TAG, max_mismatches=1):
    """
    Finds all occurrences of a specified TAG within a protein sequence allowing for a given number of mismatches.

    Args:
        protein (str): The protein sequence in which to search.
        TAG (str): The TAG sequence to find.
        max_mismatches (int): Maximum number of mismatches allowed.

    Returns:
        list: A list of indices where the TAG sequence starts within the protein, considering allowed mismatches.
    """
    sub_len = len(TAG)
    if sub_len < 5:
        max_mismatches = 0
    indexes_tags = []
    for i in range(len(protein) - sub_len + 1):
        window = protein[i:i + sub_len]
        mismatches = sum(1 for x, y in zip(window, TAG) if x != y)
        if mismatches <= max_mismatches:
            indexes_tags.append(i)
    # if not tag found return None
    if not indexes_tags:
        return None
    return indexes_tags


def calculate_codon_elongation_rates( rna, global_elongation_rate=10,remove_last_stop_codon=True):
    """
    Calculate the elongation rates for each codon in an RNA sequence based on global elongation rate and codon usage.

    Args:
        rna (str): RNA sequence.
        global_elongation_rate (float): The baseline elongation rate to adjust based on codon frequency.
        codon_frequency_dict (dict): A dictionary mapping codons to their frequency values.

    Returns:
        np.array: An array of elongation rates for each codon in the RNA sequence.
    """
    stop_codons = ['UAA', 'UAG', 'UGA']
    #average_codon_velocity = np.mean(list(codon_frequency_dict.values()))
    average_codon_frequency = np.mean([freq for codon, freq in codon_frequency_dict.items() if codon not in stop_codons])
    codon_frequency_in_gene = np.array([codon_frequency_dict[rna[i:i+3]] for i in range(0, len(rna), 3)])
    codon_frequency_normalized = codon_frequency_in_gene / average_codon_frequency
    codon_elongation_rates = codon_frequency_normalized * global_elongation_rate
    if remove_last_stop_codon:
        codon_elongation_rates = codon_elongation_rates[:-1]
    return codon_elongation_rates

# The codon elongation rate should be calculated using the formal definition using the codon usage frequency for each type of aminoacid.

"""
Codon Usage by Etsuko N. Moriyama

The codon adaptation index (CAI) estimates the extent of
bias toward codons that are known to be favored in highly ex-
pressed genes (Sharp and Li, 1987a). A “relative adaptedness”
value, wi, for codon i is calculated from its relative frequency
of use in a species-specific reference set of very highly ex-
pressed genes.
where RSCUmax and X
max are the RSCU and X values for the
most frequently used codon for an amino acid. The CAI for a
gene is then defined as the geometric mean of w values for co-
dons in that gene:

where L is the number of codons in the gene excluding methionine, tryptophan, and stop codons. The CAI ranges from 0
for no bias (all synonymous codons are used equally) to 1 for the strongest bias (only optimal codons are used).

"""



def read_sequence(seq, min_protein_length=20, TAG='YPYDVPDYA', PAUSE_SEQUENCE=None):
    """
    Reads a DNA sequence, translates it to protein, searches for ORFs, TAG sequences, and calculates codon elongation rates.

    Args:
        seq (str or pathlib.PurePath): DNA sequence or path to a file containing the DNA sequence.
        min_protein_length (int): Minimum length of protein for ORFs to be considered.
        TAG (str or list): TAG sequence(s) to find within the protein.
        global_elongation_rate (float): Baseline global elongation rate for codon usage calculations.
        codon_frequency_dict (dict): Codon frequency dictionary for elongation rate calculations.

    Returns:
        tuple: Tuple containing:
               - protein sequence (str),
               - RNA sequence (str),
               - DNA sequence (str),
               - list of TAG indices (list),
               - codon elongation rates (np.array).
    """
    # Ensure seqrecord exists for both string and file input
    if isinstance(seq, str):
        # Build a SeqRecord directly from the input string
        seq = Seq(seq)
        seqrecord = SeqRecord(seq, id="input_sequence")
    elif isinstance(seq, pathlib.PurePath):
        # Read a SeqRecord from file
        seqrecord = snapgene_file_to_seqrecord(seq)
        seq = Seq(str(seqrecord.seq))
    orfs = []
    graphic_features = convert_features(seqrecord)
    # Check both strands and three frames each
    for strand, nuc in [(+1, seq), (-1, seq.reverse_complement())]:
        for frame in range(3):
            length = 3 * ((len(nuc)-frame) // 3)  # Adjust length to complete codons
            trans = nuc[frame:frame+length].translate(to_stop=False)
            proteins = trans.split("*")
            pos = 0
            for protein in proteins:
                start_index = protein.find('M')
                if start_index != -1:  # Ensure 'M' is found
                    orf = protein[start_index:]
                    if len(orf) >= min_protein_length:
                        start_pos = frame + (pos + start_index) * 3
                        end_pos = start_pos + len(orf) * 3 + 3
                        orf_dna = nuc[start_pos:end_pos]
                        orfs.append((str(orf), str(orf_dna)))
                pos += len(protein) + 1

    if isinstance(TAG, list):
        orfs = [(orf, dna) for orf, dna in orfs if TAG[0] in orf]
    else:
        orfs = [(orf, dna) for orf, dna in orfs if TAG in orf]
    # return two variables the protein and the dna sequence
    protein, dna = orfs[0] if orfs else (None, None)
    rna = dna.upper().replace('T', 'U')

    # if TAG is a list of tags calculate the indexes of the tags for each tag
    if isinstance(TAG, list):
        indexes_tags = [find_TAG_location(protein, TAG=tag) for tag in TAG]
    else:
        indexes_tags = find_TAG_location(protein, TAG=TAG)
    if not indexes_tags:
        print('No HA tag found in the protein sequence.')

    # detect if pause sequences are in the protein sequence
    if PAUSE_SEQUENCE:
        index_pauses = find_TAG_location(protein, PAUSE_SEQUENCE)[0]
    else:
        index_pauses = None

    return protein, rna, dna, indexes_tags, index_pauses, seqrecord, graphic_features




def create_probe_vector(tag_positions, gene_length, efficiency=1):
    """
    Create a probe vector based on specified tag positions.

    Parameters:
    - tag_positions: An array of integers representing the positions on the gene where the tagging starts.
    - gene_length: The total length of the gene.

    Returns:
    - probe_vector: A numpy array where positions from each tag onward are incremented by 1.
    """
    probe_vector = np.zeros(gene_length)
    for tag in tag_positions:
        if tag < gene_length:  # Ensure the tag position is within the gene length
            if np.random.rand() <= efficiency:  # Apply efficiency check
                probe_vector[tag:] += 1 
    return probe_vector



def read_gene_sequence_return_probes(gene_sequence, min_protein_length=50, list_tag_sequences=[HA_TAG]):
    protein, rna, _, indexes_tags, _, _  = read_sequence(seq=gene_sequence, min_protein_length=min_protein_length,TAG=list_tag_sequences)
    gene_length = len(protein)+1
    tag_positions_first_probe_vector = indexes_tags[0]
    first_probe_position_vector = create_probe_vector(tag_positions_first_probe_vector, gene_length)
    tag_positions_second_probe_vector = indexes_tags[1] if len(indexes_tags) > 1 else None
    first_probe_position_vector = create_probe_vector(tag_positions_first_probe_vector, gene_length)
    second_probe_position_vector = create_probe_vector(tag_positions_second_probe_vector, gene_length) if tag_positions_second_probe_vector is not None else None
    return protein, rna, gene_length, first_probe_position_vector, second_probe_position_vector


def read_gene_sequence(file_path, TAG_list, PAUSE_SEQUENCE=None):
    protein, rna, _, indexes_tags, indexes_pause, seq_record, graphic_features = read_sequence(
        seq=file_path, TAG=TAG_list, PAUSE_SEQUENCE=PAUSE_SEQUENCE, min_protein_length=50
    )
    plasmid_figure = plot_plasmid(seq_record, graphic_features, figure_width=25, figure_height=3)
    gene_length = len(protein) + 1  # adding 1 to account for the stop codon
    probe_data = {}
    for i, (tag, tag_positions) in enumerate(zip(TAG_list, indexes_tags)):
        probe_vector = create_probe_vector(tag_positions, gene_length)
        probe_data[f'tag_{i}'] = {
            'tag_sequence': tag,
            'positions': tag_positions,
            'position_cumulative_vector': probe_vector
        }
    return {
        "protein": protein,
        "rna": rna,
        "gene_length": gene_length,
        "probe_data": probe_data,  # Dictionary with all probe information
        "plasmid_figure": plasmid_figure,
        "seq_record": seq_record,
        "graphic_features": graphic_features,
        "num_probes": len(indexes_tags),
        "pause_indexes": indexes_pause
    }


# Function to get feature color
def get_feature_color(feature_type: str, qualifiers) -> str:
    if 'note' in qualifiers:
        for note in qualifiers['note']:
            if note.lower().startswith('color:'):
                return note.split(':')[1].strip()
    color_dict = {
        'cds': '#57B956',               
        'promoter': '#ff0000',            
        'origin_of_replication': '#EB5559',  
        'rep_origin': '#C4B07B',          
    }
    return color_dict.get(feature_type.lower(), '#cccccc')  # Default to gray

# Function to convert features
def convert_features(seq_record: SeqRecord) -> List[GraphicFeature]:
    graphic_features = []
    for feature in seq_record.features:
        feature_type = feature.type.lower()
        list_no_plot = ['G67A']
        if feature_type in ['rep_origin', 'cds', 'promoter']: 
            #if feature.qualifiers['label'] in list_no_plot: # and feature.qualifiers['label'] not in list_no_plot:
            if all(sub not in feature.qualifiers['label'] for sub in list_no_plot):
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand = feature.location.strand
                qualifiers = feature.qualifiers
                # Get a descriptive label
                label =  qualifiers['label'] # get_label_from_qualifiers(qualifiers, feature_type)
                # Get the feature color
                color = get_feature_color(feature_type, qualifiers)
                # Create the GraphicFeature
                graphic_feature = GraphicFeature(
                    start=start,
                    end=end,
                    strand=strand,
                    color=color,
                    label=label,
                    fontdict = { 'weight': 'bold', 'family':'Helvetica', 'fontsize': 8}
                )
                graphic_features.append(graphic_feature)
    return graphic_features

# Function to plot plasmid
def plot_plasmid(seq_record: SeqRecord, graphic_features: List[GraphicFeature], figure_width: int = 20, figure_height: int = 5) -> plt.Figure:
    graphic_record = GraphicRecord( # CircularGraphicRecord
        sequence_length=len(seq_record.seq),
        features=graphic_features,
    )
    ax, _ = graphic_record.plot(figure_width=figure_width, figure_height=figure_height, strand_in_label_threshold=2)
    ax.set_title('Plasmid Map')
    plt.show()
    return ax.figure




def TASEP_ODE(p, t, ki, k_elongation, k_termination):
    """
    ODE system for a simplified TASEP-like model (deterministic).
    p: occupancy array along the gene (length = number of codons).
    dpdt is computed using constant initiation, codon-specific elongation, and
    constant termination.
    """
    N = len(p)  # Total number of codon positions
    dpdt = np.zeros(N)
    # Handle each codon
    dpdt[0] = ki - k_elongation[0] * p[0]  # First codon
    for i in range(1, N - 1):
        dpdt[i] = k_elongation[i - 1] * p[i - 1] - k_elongation[i] * p[i]
    dpdt[N - 1] = k_elongation[N - 2] * p[N - 2] - k_termination * p[N - 1]  # Last codon
    return dpdt

def simulate_TASEP_ODE(
    ki,
    ke,
    gene_length,
    t_max,
    first_probe_position_vector,
    second_probe_position_vector=None,
    burnin_time=0,
    time_interval_in_seconds=1.0,
    pause_location=None, 
    pause_elongation_rate=None,
):
    """
    Solves a simplified TASEP ODE system deterministically from t=0 to t=t_max,
    with optional burn-in time removed from the final output.

    Parameters
    ----------
    ki : float
        Initiation rate.
    ke : float or array-like
        Elongation rate (if scalar) or per-codon array (length = gene_length).
    gene_length : int
        Total number of codons (for the ODE system).
    t_max : float
        Maximum simulation time (in the same units as ki, ke, etc.).
    first_probe_position_vector : np.ndarray
        (gene_length,) array indicating which codons are covered by the first probe.
    second_probe_position_vector : np.ndarray or None
        Optional second probe array (same length).
    burnin_time : float
        If > 0, an initial period from 0..burnin_time is "discarded" from the final signal.
    time_interval_in_seconds : float
        Step size for storing the ODE solution. Default = 1.0.

    Returns
    -------
    intensity_vector_first_signal_ode : np.ndarray
        1D array of length #timesteps (minus burnin frames) for the first probe.
    intensity_vector_second_signal_ode : np.ndarray or None
        Similarly for the second probe if provided, else None.
    """
    # 1) If burnin_time is used, shift t_max accordingly for the solver
    if burnin_time > 0:
        t_max += burnin_time

    # 2) Build the time array
    t = np.arange(0, t_max, time_interval_in_seconds)

    # 3) Construct codon elongation rates
    if isinstance(ke, (int, float)):
        # constant elongation for all codons
        k_elongation = np.full(gene_length, ke, dtype=float)
    else:
        # user-supplied array (must match gene_length)
        k_elongation = np.array(ke, dtype=float)

    if pause_location is not None and pause_elongation_rate is not None:
        if 0 <= pause_location < gene_length:
            k_elongation[pause_location] = pause_elongation_rate  # Set elongation rate at pause location
        else:
            raise ValueError("pause_location must be within the range of gene_length.")


    # 4) For simplicity, define termination rate as mean of the elongation rates (or up to you)
    k_termination = np.mean(k_elongation)

    # 5) Initial occupancy (all zeros)
    p0 = np.zeros(gene_length, dtype=float)

    # 6) Solve ODE
    p_solution = odeint(
        TASEP_ODE,
        p0,
        t,
        args=(ki, k_elongation, k_termination)
    )
    intensity_vector_first_signal_ode = np.dot(first_probe_position_vector, p_solution.T)
    if second_probe_position_vector is not None:
        intensity_vector_second_signal_ode = np.dot(second_probe_position_vector, p_solution.T)
    else:
        intensity_vector_second_signal_ode = None

    # 8) Remove burnin frames, if applicable
    if burnin_time > 0:
        burnin_index = int(burnin_time / time_interval_in_seconds)
        intensity_vector_first_signal_ode = intensity_vector_first_signal_ode[burnin_index:]
        if second_probe_position_vector is not None:
            intensity_vector_second_signal_ode = intensity_vector_second_signal_ode[burnin_index:]
    else:
        burnin_index = 0

    return intensity_vector_first_signal_ode, intensity_vector_second_signal_ode



# -----------------------------------------------------------------------------
# Per-event folding-delay gating (ribosome-resolved)
# -----------------------------------------------------------------------------
def _probe_start_codon_index(probe_vec):
    """
    Return the 1-based codon index where a probe becomes active (first nonzero entry).
    If probe is None or has no nonzero entries, return None.
    """
    if probe_vec is None:
        return None
    probe_vec = np.asarray(probe_vec)
    nz = np.flatnonzero(probe_vec > 0)
    return int(nz[0] + 1) if nz.size > 0 else None


def _per_event_second_signal_for_rep(
    ribo_traj_rep,              # shape (n_rib, T) int codon positions; 0 when absent
    t_array,                    # shape (T,)
    first_probe_position_vector,
    second_probe_position_vector,
    folding_delay_seconds
):
    """
    Build a per-event delayed second signal by gating EACH ribosome's contribution:
      - the ribosome must have reached the first probe (if provided)
      - the ribosome must have reached the second probe
      - folding_delay_seconds elapsed after second-probe arrival
    After the gate opens, that ribosome contributes second_probe_vector[pos-1] at each timepoint.
    """
    T = t_array.shape[0]
    if ribo_traj_rep is None or ribo_traj_rep.size == 0:
        return np.zeros(T, dtype=np.float64)

    ribo_traj_rep = np.asarray(ribo_traj_rep, dtype=np.int64)
    if ribo_traj_rep.ndim == 1:
        ribo_traj_rep = ribo_traj_rep[None, :]

    first_start  = _probe_start_codon_index(first_probe_position_vector)
    second_start = _probe_start_codon_index(second_probe_position_vector)
    if second_start is None:
        return np.zeros(T, dtype=np.float64)

    second_vec = np.asarray(second_probe_position_vector, dtype=np.float64)
    out = np.zeros(T, dtype=np.float64)

    for r in range(ribo_traj_rep.shape[0]):
        pos = ribo_traj_rep[r, :]  # integer codon index (1..L), 0 when absent

        # when does this ribosome first reach the first probe?
        if first_start is None:
            idx_first = 0
        else:
            hits_first = np.flatnonzero(pos >= first_start)
            idx_first = int(hits_first[0]) if hits_first.size > 0 else None

        # when does it first reach the second probe?
        hits_second = np.flatnonzero(pos >= second_start)
        if hits_second.size == 0:
            continue
        idx_second = int(hits_second[0])

        # enforce "first before second" if first probe exists
        if (idx_first is not None) and (idx_second < idx_first):
            continue

        # gate time = first time at/after (t_second + folding_delay)
        t_gate  = t_array[idx_second] + float(folding_delay_seconds)
        idx_gate = int(np.searchsorted(t_array, t_gate, side='left'))
        if idx_gate >= T:
            continue

        # contribution = second_probe_value at current codon (0 if off-gene)
        contrib = np.zeros(T, dtype=np.float64)
        valid = pos > 0
        if np.any(valid):
            contrib[valid] = second_vec[(pos[valid] - 1).astype(np.int64)]
        contrib[:idx_gate] = 0.0

        out += contrib

    return out


def apply_per_event_folding_delay(
    list_ribosome_trajectories,     # list of (n_rib, T) int arrays, length = number_repetitions
    t_array,                        # shape (T,)
    first_probe_position_vector,
    second_probe_position_vector,
    folding_delay_seconds,
    burnin_time=0.0
):
    """
    Build matrix_intensity_second_signal_RT_delayed using per-event gating.
    Returns (R, T') where T' = T minus burn-in frames if burnin_time>0; else T.
    """
    if (list_ribosome_trajectories is None) or (len(list_ribosome_trajectories) == 0):
        return None

    out_mat = []
    for ribo_traj in list_ribosome_trajectories:
        sig = _per_event_second_signal_for_rep(
            ribo_traj, t_array,
            first_probe_position_vector,
            second_probe_position_vector,
            folding_delay_seconds
        )
        out_mat.append(sig)

    out_mat = np.vstack(out_mat) if len(out_mat) > 0 else None

    if (out_mat is not None) and (burnin_time > 0) and (t_array.size > 1):
        idx_burn = int(burnin_time / float(t_array[1] - t_array[0]))
        out_mat = out_mat[:, idx_burn:]

    return out_mat

def _delay_signal_fractional(signal_1d, delay_frames_float):
    n = len(signal_1d)
    t = np.arange(n, dtype=float)
    return np.interp(t - delay_frames_float, t, signal_1d, left=0.0, right=0.0)


# -----------------------------------------------------------------------------
# Numba-accelerated SSA simulation (internal function)
# Always returns a tuple:
#   (ribosome_trajectories, occupancy_output, intensity_first_signal)
# In full-output mode, intensity_first_signal is empty.
# In fast-output mode, ribosome_trajectories and occupancy_output are empty.
# -----------------------------------------------------------------------------





@njit
def TASEP_SSA_numba(k, t_array, timePerturbationApplication, evaluatingInhibitor,
                    evaluatingFRAP, inhibitor_effectiveness, constant_elongation_rate,
                    fast_output, first_probe_position_vector):
    """
    Numba-accelerated TASEP SSA simulation.
    
    Parameters
    ----------
    k : 1D np.array of float64, shape (L+2,)
         [k_bind, k_1, k_2, ..., k_L, k_termination].
    t_array : 1D np.array of float64
         Recording times.
    timePerturbationApplication : float64
         Time when an inhibitor is applied.
    evaluatingInhibitor : int32 (0 or 1)
         Whether inhibitor is active.
    evaluatingFRAP : int32 (0 or 1)
         Whether FRAP is active.
    inhibitor_effectiveness : float64
         Inhibition power in percent (e.g., 100 means full inhibition of initiation,
         0.1 means 0.1% inhibition).
    constant_elongation_rate : float64
         If >= 0, use this as the uniform elongation rate; if negative then use codon‐dependent rates from k[1:-1].
    fast_output : int32 (0 or 1)
         If 1, only compute first-probe intensity.
    first_probe_position_vector : 1D np.array of float64
         Probe coverage vector (length gene_length). If not used, pass an array of length 0.
         
    Returns
    -------
    A 3-tuple:
      ribosome_trajectories : 2D np.array of int64, shape (n_ribosomes, num_timepoints) 
                              (empty if fast_output==1)
      occupancy_output    : 2D np.array of float64, shape ((gene_length+2), num_timepoints)
                              (empty if fast_output==1)
      intensity_first_signal : 1D np.array of float64, length num_timepoints
                              (empty if fast_output==0)
    """
    exclusion = 9  # ribosome footprint
    k_bind = k[0]
    k_term = k[k.shape[0]-1]
    gene_length = k.shape[0] - 2

    use_constant = (constant_elongation_rate >= 0)
    if not use_constant:
        k_elongation = k[1:k.shape[0]-1]  # site-specific rates
    # else: we'll use constant_elongation_rate

    t = t_array[0]
    t_final = t_array[t_array.shape[0]-1]
    num_timepoints = t_array.shape[0]

    # Pre-allocate outputs.
    if fast_output == 1:
        intensity_first_signal = np.zeros(num_timepoints, dtype=np.float64)
        # For fast output, we return empty arrays for the other two.
        ribosome_trajectories = np.empty((0, num_timepoints), dtype=np.int64)
        occupancy_output = np.empty((0, num_timepoints), dtype=np.float64)
    else:
        occupancy_output = np.zeros((gene_length + 2, num_timepoints), dtype=np.float64)
        # We'll collect ribosome trajectories in a typed list.
        ribosome_positions_list = TypedList.empty_list(types.float64[:])
        intensity_first_signal = np.empty(0, dtype=np.float64)  # not used in full output

    # Initialize dynamic lists (all explicitly typed).
    active_positions = TypedList.empty_list(types.int64)   # positions (1-indexed)
    initiation_times = TypedList.empty_list(types.float64)   # initiation times
    trajectory_indices = TypedList.empty_list(types.int64)   # indices into ribosome_positions_list

    iter_time_idx = 0

    # Main SSA loop.
    while t < t_final:
        # (A) Inhibitor: Only affect initiation.
        if (t >= timePerturbationApplication) and (evaluatingInhibitor == 1):
            # Convert inhibitor_effectiveness percentage into a multiplier.
            # For example, inhibitor_effectiveness=100 --> multiplier = 0 (full inhibition),
            # inhibitor_effectiveness=0.1 --> multiplier = 0.999 (0.1% inhibition).
            current_inhib_factor = 1.0 - inhibitor_effectiveness / 100.0
        else:
            current_inhib_factor = 1.0

        # (B) FRAP.
        if (evaluatingFRAP == 1) and (t >= timePerturbationApplication) and (t <= timePerturbationApplication + 10.0):
            active_positions = TypedList.empty_list(types.int64)
            initiation_times = TypedList.empty_list(types.float64)
            trajectory_indices = TypedList.empty_list(types.int64)
            if fast_output == 0:
                for i in range(len(ribosome_positions_list)):
                    for j in range(iter_time_idx, num_timepoints):
                        ribosome_positions_list[i][j] = np.nan

        # (C) Build propensities.
        n_rib = len(active_positions)
        # Initiation propensity is inhibited.
        if n_rib == 0 or (n_rib > 0 and active_positions[0] > exclusion):
            init_prop = k_bind * current_inhib_factor
        else:
            init_prop = 0.0

        # Elongation: inhibitor is NOT applied.
        elong_props = TypedList.empty_list(types.float64)
        elong_indices = TypedList.empty_list(types.int64)
        for i in range(n_rib):
            pos = active_positions[i]
            if pos <= gene_length - 1:
                if i == n_rib - 1:
                    can_elongate = True
                else:
                    can_elongate = ((pos + exclusion) < active_positions[i+1])
                if can_elongate:
                    if use_constant:
                        elong_rate = constant_elongation_rate
                    else:
                        if (pos >= 1) and (pos <= gene_length):
                            elong_rate = k_elongation[pos-1]
                        else:
                            elong_rate = 0.0
                    if elong_rate > 0:
                        elong_props.append(elong_rate)  # no inhibitor factor here!
                        elong_indices.append(i)
        # Termination: inhibitor is NOT applied.
        term_props = TypedList.empty_list(types.float64)
        term_indices = TypedList.empty_list(types.int64)
        for i in range(n_rib):
            pos = active_positions[i]
            if pos >= gene_length:
                term_props.append(k_term)  # no inhibitor factor
                term_indices.append(i)

        total_events = 0
        if init_prop > 0:
            total_events += 1
        total_events += len(elong_props)
        total_events += len(term_props)
        prop_arr = np.empty(total_events, dtype=np.float64)
        reaction_type = np.empty(total_events, dtype=np.int32)  # 0: initiation, 1: elongation, 2: termination.
        reaction_index = np.empty(total_events, dtype=np.int32)

        event_counter = 0
        if init_prop > 0:
            prop_arr[event_counter] = init_prop
            reaction_type[event_counter] = 0
            reaction_index[event_counter] = -1
            event_counter += 1
        for i in range(len(elong_props)):
            prop_arr[event_counter] = elong_props[i]
            reaction_type[event_counter] = 1
            reaction_index[event_counter] = elong_indices[i]
            event_counter += 1
        for i in range(len(term_props)):
            prop_arr[event_counter] = term_props[i]
            reaction_type[event_counter] = 2
            reaction_index[event_counter] = term_indices[i]
            event_counter += 1

        sum_prop = prop_arr.sum()
        if sum_prop <= 0:
            t = t_final
        else:
            tau = -np.log(np.random.rand()) / sum_prop
            if (evaluatingInhibitor == 1) and (t < timePerturbationApplication) and ((t + tau) > timePerturbationApplication):
                t = timePerturbationApplication
            else:
                t += tau
                r2 = sum_prop * np.random.rand()
                cumul = 0.0
                i_rxn = 0
                while i_rxn < total_events:
                    cumul += prop_arr[i_rxn]
                    if cumul >= r2:
                        break
                    i_rxn += 1
                r_type = reaction_type[i_rxn]
                r_idx = reaction_index[i_rxn]
                if r_type == 0:
                    # Initiation.
                    new_pos = 1
                    insert_idx = 0
                    while insert_idx < n_rib and active_positions[insert_idx] < new_pos:
                        insert_idx += 1
                    active_positions.insert(insert_idx, new_pos)
                    initiation_times.insert(insert_idx, t)
                    if fast_output == 0:
                        new_row = np.full(num_timepoints, np.nan, dtype=np.float64)
                        ribosome_positions_list.append(new_row)
                        trajectory_indices.insert(insert_idx, len(ribosome_positions_list)-1)
                elif r_type == 1:
                    active_positions[r_idx] = active_positions[r_idx] + 1
                    j = r_idx
                    while (j < len(active_positions)-1) and (active_positions[j] > active_positions[j+1]):
                        # Swap positions.
                        tmp = active_positions[j]
                        active_positions[j] = active_positions[j+1]
                        active_positions[j+1] = tmp
                        # Swap initiation times.
                        tmp = initiation_times[j]
                        initiation_times[j] = initiation_times[j+1]
                        initiation_times[j+1] = tmp
                        if fast_output == 0:
                            tmp = trajectory_indices[j]
                            trajectory_indices[j] = trajectory_indices[j+1]
                            trajectory_indices[j+1] = tmp
                        j += 1
                elif r_type == 2:
                    for j in range(r_idx, len(active_positions)-1):
                        active_positions[j] = active_positions[j+1]
                        initiation_times[j] = initiation_times[j+1]
                        if fast_output == 0:
                            trajectory_indices[j] = trajectory_indices[j+1]
                    active_positions.pop()
                    initiation_times.pop()
                    if fast_output == 0:
                        trajectory_indices.pop()
        # (F) Record state.
        while iter_time_idx < num_timepoints and t >= t_array[iter_time_idx]:
            if fast_output == 0:
                occ_vec = np.zeros(gene_length, dtype=np.float64)
                for i in range(len(active_positions)):
                    pos = active_positions[i]
                    if pos >= 1 and pos <= gene_length:
                        occ_vec[pos-1] = 1.0
                for j in range(gene_length):
                    occupancy_output[j+1, iter_time_idx] = occ_vec[j]
                for i in range(len(trajectory_indices)):
                    row_idx = trajectory_indices[i]
                    p = active_positions[i]
                    if (p >= 1) and (p <= gene_length):
                        if t_array[iter_time_idx] >= initiation_times[i]:
                            ribosome_positions_list[row_idx][iter_time_idx] = p
                        else:
                            ribosome_positions_list[row_idx][iter_time_idx] = np.nan
                    else:
                        ribosome_positions_list[row_idx][iter_time_idx] = np.nan
            else:
                if first_probe_position_vector.shape[0] > 0:
                    sum_occ = 0.0
                    for i in range(len(active_positions)):
                        pos = active_positions[i]
                        if (pos >= 1) and (pos <= gene_length):
                            sum_occ += first_probe_position_vector[pos-1]
                    intensity_first_signal[iter_time_idx] = sum_occ
            iter_time_idx += 1

    # End of main loop.
    if fast_output == 1:
        return np.empty((0, num_timepoints), dtype=np.int64), np.empty((0, num_timepoints), dtype=np.float64), intensity_first_signal
    else:
        if len(ribosome_positions_list) > 0:
            n_rib = len(ribosome_positions_list)
            ribosome_trajectories = np.empty((n_rib, num_timepoints), dtype=np.float64)
            for i in range(n_rib):
                for j in range(num_timepoints):
                    ribosome_trajectories[i, j] = ribosome_positions_list[i][j]
            for i in range(n_rib):
                for j in range(num_timepoints):
                    if np.isnan(ribosome_trajectories[i, j]):
                        ribosome_trajectories[i, j] = 0.0
            ribosome_trajectories = ribosome_trajectories.astype(np.int64)
        else:
            ribosome_trajectories = np.zeros((0, num_timepoints), dtype=np.int64)
        return ribosome_trajectories, occupancy_output, np.empty(0, dtype=np.float64)
    
# -----------------------------------------------------------------------------
# Wrapper (unchanged interface)
# -----------------------------------------------------------------------------
def TASEP_SSA(k, t_array, timePerturbationApplication=0, evaluatingInhibitor=0, evaluatingFRAP=0,
              inhibitor_effectiveness=1.0, constant_elongation_rate=None, fast_output=False,
              first_probe_position_vector=None):
    """
    Wrapper for the Numba-accelerated TASEP_SSA_numba.
    For constant_elongation_rate, pass a negative value (e.g. -1.0) to use site-specific rates.
    Boolean flags should be passed as 0 or 1.
    """
    k = np.asarray(k, dtype=np.float64)
    t_array = np.asarray(t_array, dtype=np.float64)
    if first_probe_position_vector is None:
        first_probe_position_vector = np.empty(0, dtype=np.float64)
    else:
        first_probe_position_vector = np.asarray(first_probe_position_vector, dtype=np.float64)
    # Call the numba function.
    rt = TASEP_SSA_numba(k, t_array, timePerturbationApplication, int(evaluatingInhibitor),
                         int(evaluatingFRAP), inhibitor_effectiveness,
                         constant_elongation_rate if constant_elongation_rate is not None else -1.0,
                         int(fast_output), first_probe_position_vector)
    # Unpack return.
    ribo_traj, occ_out, intensity_first_signal = rt
    if fast_output:
        return intensity_first_signal
    else:
        return ribo_traj, occ_out

def simulate_TASEP_SSA(ki, ke, gene_length, t_max, time_interval_in_seconds=1, number_repetitions=1,
                       first_probe_position_vector=None, second_probe_position_vector=None,
                       timePerturbationApplication=0, evaluatingInhibitor=0, evaluatingFRAP=0,
                       n_jobs=-1, folding_delay=0, burnin_time=0, inhibitor_effectiveness=0,
                        efficiency_list=None, constant_elongation_rate=None,
                       pause_location=None, pause_elongation_rate=None,
                       fast_output=False, batch_size='auto', gate_by_first_signal_per_event=False):
    """
    Parallel wrapper for TASEP_SSA.
    Returns a tuple:
      (list_ribosome_trajectories, list_occupancy_output, matrix_intensity_first_signal_RT, matrix_intensity_second_signal_RT_delayed)
    """
    if burnin_time > 0:
        timePerturbationApplication = (timePerturbationApplication or 0) + burnin_time
        t_max += burnin_time

    t_array = np.arange(0, t_max, time_interval_in_seconds)

    if isinstance(ke, (int, float)):
        k_elongation = np.full(gene_length - 2, ke, dtype=np.float64)
    else:
        k_elongation = np.array(ke, dtype=np.float64)
    k_termination = k_elongation[-1]
    k_full = np.concatenate(([ki], k_elongation, [k_termination])).astype(np.float64)

    if pause_location is not None and pause_elongation_rate is not None:
        if 0 <= pause_location < gene_length:
            k_full[pause_location] = pause_elongation_rate  # Set elongation rate at pause location
        else:
            raise ValueError("pause_location must be within the range of gene_length.")
    
    args_list = []
    for _ in range(number_repetitions):
        args_list.append((k_full, t_array, timePerturbationApplication, evaluatingInhibitor,
                          evaluatingFRAP, inhibitor_effectiveness, constant_elongation_rate,
                          fast_output, first_probe_position_vector))

    def run_single_simulation(args):
        result = TASEP_SSA(*args)
        if fast_output:
            return {'intensity_first_signal': result,
                    'ribosome_trajectories': None,
                    'occupancy_output': None}
        else:
            ribo_traj, occ_out = result
            res = {'ribosome_trajectories': ribo_traj,
                   'occupancy_output': occ_out}
            if first_probe_position_vector is not None and first_probe_position_vector.size > 0:
                occ_slice = occ_out[1:-1, :]
                if efficiency_list is not None and len(efficiency_list) > 0:
                    first_probe_position_detection = np.where(np.diff(first_probe_position_vector) != 0)[0] + 1
                    first_probe_position_vector_eff = create_probe_vector(first_probe_position_detection, gene_length-1, efficiency=(efficiency_list[0] if efficiency_list is not None else 1.0))
                else:
                    first_probe_position_vector_eff = first_probe_position_vector
                first_int = np.sum(first_probe_position_vector_eff * occ_slice.T, axis=1)
                res['intensity_first_signal'] = first_int
            else:
                res['intensity_first_signal'] = None
            if second_probe_position_vector is not None:
                occ_slice = occ_out[1:-1, :]
                if efficiency_list is not None and len(efficiency_list) > 1:
                    second_probe_position_detection = np.where(np.diff(second_probe_position_vector) != 0)[0] + 1
                    second_probe_position_vector_eff = create_probe_vector(second_probe_position_detection, gene_length-1, efficiency=(efficiency_list[1] if efficiency_list is not None else 1.0))
                else:
                    second_probe_position_vector_eff = second_probe_position_vector
                second_int = np.sum(second_probe_position_vector_eff * occ_slice.T, axis=1)
                res['intensity_second_signal'] = second_int
            else:
                res['intensity_second_signal'] = None
            return res

    try:
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(run_single_simulation)(args) for args in args_list
        )
    except Exception:
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(run_single_simulation)(args) for args in args_list
        )

    list_ribosome_trajectories = [r['ribosome_trajectories'] for r in results]
    list_occupancy_output = [r['occupancy_output'] for r in results]
    list_first_signal = [r.get('intensity_first_signal', None) for r in results]
    list_second_signal = [r.get('intensity_second_signal', None) for r in results]

    matrix_intensity_first_signal_RT = (np.array(list_first_signal)
                                         if all(x is not None for x in list_first_signal)
                                         else None)
    matrix_intensity_second_signal_RT = (np.array(list_second_signal)
                                          if all(x is not None for x in list_second_signal)
                                          else None)

    # if folding_delay > 0 and matrix_intensity_second_signal_RT is not None:
    #     matrix_intensity_second_signal_RT_delayed = np.zeros_like(matrix_intensity_second_signal_RT)
    #     delay_frames = int(folding_delay / time_interval_in_seconds)
    #     for i_rep in range(number_repetitions):
    #         matrix_intensity_second_signal_RT_delayed[i_rep, :] = delay_signal(
    #             matrix_intensity_second_signal_RT[i_rep, :], delay_frames
    #         )
    # else:
    #     matrix_intensity_second_signal_RT_delayed = matrix_intensity_second_signal_RT

    if folding_delay > 0 and matrix_intensity_second_signal_RT is not None:
        if (not fast_output) and gate_by_first_signal_per_event:
            # Per-event gating using ribosome trajectories (no pre-trim; burn-in is applied later)
            matrix_intensity_second_signal_RT_delayed = apply_per_event_folding_delay(
                list_ribosome_trajectories=list_ribosome_trajectories,
                t_array=t_array,
                first_probe_position_vector=first_probe_position_vector,
                second_probe_position_vector=second_probe_position_vector,
                folding_delay_seconds=folding_delay,
                burnin_time=0.0
            )
        else:
            # Fallback: global fractional shift of the aggregated second signal
            delay_frames_float = float(folding_delay) / float(time_interval_in_seconds)
            matrix_intensity_second_signal_RT_delayed = np.zeros_like(matrix_intensity_second_signal_RT)
            for i_rep in range(number_repetitions):
                matrix_intensity_second_signal_RT_delayed[i_rep, :] = _delay_signal_fractional(
                    matrix_intensity_second_signal_RT[i_rep, :], delay_frames_float
                )
    else:
        matrix_intensity_second_signal_RT_delayed = matrix_intensity_second_signal_RT


    if burnin_time > 0:
        idx_burnin = int(burnin_time / time_interval_in_seconds)
        if matrix_intensity_first_signal_RT is not None:
            matrix_intensity_first_signal_RT = matrix_intensity_first_signal_RT[:, idx_burnin:]
        if matrix_intensity_second_signal_RT_delayed is not None:
            matrix_intensity_second_signal_RT_delayed = matrix_intensity_second_signal_RT_delayed[:, idx_burnin:]
        if not fast_output:
            list_ribosome_trajectories = [traj[:, idx_burnin:] if traj is not None else None for traj in list_ribosome_trajectories]
            list_occupancy_output = [occ[:, idx_burnin:] if occ is not None else None for occ in list_occupancy_output]

    return (list_ribosome_trajectories,
            list_occupancy_output,
            matrix_intensity_first_signal_RT,
            matrix_intensity_second_signal_RT_delayed)



def plot_trajectories(matrix_intensity_first_signal_RT, intensity_vector_first_signal_ode, time_array, number_repetitions, plot_color = 'orangered'):
    # --- Set fonts and background as before ---
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    # --- Determine the global intensity range from both datasets ---
    global_min = min(matrix_intensity_first_signal_RT.min(), intensity_vector_first_signal_ode.min())
    global_max = max(matrix_intensity_first_signal_RT.max(), intensity_vector_first_signal_ode.max())

    # --- Create subplots: left for trajectories, right for histogram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3), gridspec_kw={'width_ratios': [4, 1]})

    # --- Left Plot: Trajectories ---
    for i in range(number_repetitions):
        if i == 0:
            ax1.plot(time_array, matrix_intensity_first_signal_RT[i, :],
                    label='SSA', color=plot_color, alpha=1, linewidth=2)
        else:
            ax1.plot(time_array, matrix_intensity_first_signal_RT[i, :],
                    color=plot_color, alpha=0.1, linewidth=0.4)
    ax1.plot(time_array, intensity_vector_first_signal_ode, label='ODE', color='k', linewidth=3)

    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=20)
    ax1.set_ylim(global_min, global_max)

    # Set the axes frame with a distinct black border for ax1:
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    # Place the legend in the upper right corner with a black border
    legend1 = ax1.legend(loc='upper right', fontsize=14)
    legend1.get_frame().set_edgecolor('black')
    legend1.get_frame().set_linewidth(1.5)

    ax1.grid(False)  # Remove grid lines
    ax1.tick_params(axis='both', which='major', labelsize=16)


    # --- Right Plot: Horizontal Histogram of SSA Trajectories ---
    # Flatten all SSA trajectory values into a single array
    ssa_values = matrix_intensity_first_signal_RT.flatten()

    ax2.hist(ssa_values, bins=100, orientation='horizontal',
            color=plot_color, alpha=0.7)
    ax2.set_xlabel('Counts', fontsize=20)
    ax2.set_ylabel('Intensity (a.u.)', fontsize=20)
    ax2.set_ylim(global_min, global_max)
    # set axis font size
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Set the axes frame with a distinct black border for ax2:
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    ax2.grid(False)  # Remove grid lines

    plt.tight_layout()
    plt.show()



def plot_dual_signal_trajectories(matrix_intensity_first_signal_RT, matrix_intensity_second_signal_RT,
                                  time_array, trajectory_index=0,
                                  colors=['forestgreen', 'indigo'],
                                  labels=['Signal 1', 'Signal 2'], smooth_window=1,
                                  normalize=True):
    """
    Plot a single trajectory for two different signals with proper NaN handling.
    
    Parameters:
    -----------
    matrix_intensity_first_signal_RT : np.ndarray
        First signal trajectories (shape: number_trajectories x time_points)
    matrix_intensity_second_signal_RT : np.ndarray  
        Second signal trajectories (shape: number_trajectories x time_points)
    time_array : np.ndarray
        Time points
    trajectory_index : int
        Which trajectory to plot (default: 0 for first trajectory)
    colors : list of str
        Colors for first and second signals
    labels : list of str
        Labels for first and second signals
    normalize : bool
        If True, normalize both signals from 0 to 1 (default: True)
    """

    # Validate trajectory index
    max_trajectories = matrix_intensity_first_signal_RT.shape[0]
    if trajectory_index >= max_trajectories:
        raise ValueError(f"trajectory_index {trajectory_index} is out of bounds. "
                        f"Available trajectories: 0-{max_trajectories-1}")

    # Extract the selected trajectories
    first_signal = matrix_intensity_first_signal_RT[trajectory_index, :].copy()
    second_signal = matrix_intensity_second_signal_RT[trajectory_index, :].copy()

    # Apply smoothing if requested
    if smooth_window > 1:
        first_signal = pd.Series(first_signal).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()
        second_signal = pd.Series(second_signal).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()

    # Check for valid data
    n_valid_first = np.sum(np.isfinite(first_signal))
    n_valid_second = np.sum(np.isfinite(second_signal))
    
    print(f"Trajectory #{trajectory_index} diagnostics:")
    print(f"  First signal: {n_valid_first}/{len(first_signal)} valid points")
    print(f"  Second signal: {n_valid_second}/{len(second_signal)} valid points")
    
    if n_valid_first < 2:
        raise ValueError(f"First signal has insufficient valid data points ({n_valid_first})")
    if n_valid_second < 2:
        raise ValueError(f"Second signal has insufficient valid data points ({n_valid_second})")
    
    # Normalize signals if requested
    if normalize:
        # First signal normalization
        first_min = np.nanmin(first_signal)
        first_max = np.nanmax(first_signal)
        
        if np.isfinite(first_min) and np.isfinite(first_max) and first_max > first_min:
            first_signal = (first_signal - first_min) / (first_max - first_min)
            print(f"  First signal normalized: [{first_min:.3f}, {first_max:.3f}] → [0, 1]")
        else:
            print(f"  WARNING: First signal not normalized (min={first_min}, max={first_max})")
            first_signal = first_signal - np.nanmean(first_signal)
        
        # Second signal normalization
        second_min = np.nanmin(second_signal)
        second_max = np.nanmax(second_signal)
        
        if np.isfinite(second_min) and np.isfinite(second_max) and second_max > second_min:
            second_signal = (second_signal - second_min) / (second_max - second_min)
            print(f"  Second signal normalized: [{second_min:.3f}, {second_max:.3f}] → [0, 1]")
        else:
            print(f"  WARNING: Second signal not normalized (min={second_min}, max={second_max})")
            second_signal = second_signal - np.nanmean(second_signal)

    # --- Create single plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))

    # --- Plot both signals (handling NaNs) ---
    # Create masks for valid data
    valid_first = np.isfinite(first_signal)
    valid_second = np.isfinite(second_signal)
    
    # Plot first signal
    ax.plot(time_array[valid_first], first_signal[valid_first], 
           label=labels[0], color=colors[0], linewidth=2, alpha=0.8, marker='o', markersize=3)
    
    # Plot second signal
    ax.plot(time_array[valid_second], second_signal[valid_second], 
           label=labels[1], color=colors[1], linewidth=2, alpha=0.8, marker='o', markersize=3)

    ax.set_xlabel('Time (s)', fontsize=20)
    if normalize:
        ax.set_ylabel('Norm. Int.', fontsize=20)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_ylabel('Intensity (a.u.)', fontsize=20)

    # Set axes frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    # Legend
    legend = ax.legend(loc='upper right', fontsize=14)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Add trajectory index and data quality to title
    ax.set_title(f'Trajectory #{trajectory_index} '
                f'({n_valid_first}/{len(first_signal)} pts first, '
                f'{n_valid_second}/{len(second_signal)} pts second)', 
                fontsize=16)

    plt.tight_layout()
    plt.show()



def plot_RibosomeMovement(RibosomePositions, IntensityVector, probePositions, SecondIntensityVector=None, second_probePositions=None, fileNameGif='temp_gif', color='red',second_color ='lime', FrameVelocity=10, timePerturbationApplication= None):
    """
    Function to plot ribosome movement and intensity over time, and generate an animation as a GIF.

    Parameters:
    - RibosomePositions: numpy array of shape (num_ribosomes, num_timepoints)
    - IntensityVector: numpy array of length num_timepoints
    - time: numpy array of time points
    - geneLength: length of the gene (scalar)
    - fileNameGif: filename for the output GIF (without extension)
    - probePositions: numpy array of probe positions along the gene
    - timePerturbationApplication: time when perturbation is applied
    - color: color to use for plotting (e.g., 'blue')
    - FrameVelocity: frames per second (int)
    """
    # Normalize IntensityVector
    time = np.arange(0, len(IntensityVector),1)
    geneLength = np.max(RibosomePositions)
    IntensityVector = IntensityVector / np.max(IntensityVector)
    if SecondIntensityVector is not None:
        SecondIntensityVector = SecondIntensityVector / np.max(SecondIntensityVector)
    maxIntensity = 1
    Max_No_Ribosomes, num_timepoints = RibosomePositions.shape


    timePoints = len(time)
    if geneLength > 1100:
        pointSize = 4.5
    else:
        pointSize = 6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4), facecolor='black',gridspec_kw={'height_ratios': [0.6, 0.4]})
    fig.subplots_adjust(hspace=0.5)
    stepSize = 5

    # Prepare the frames for animation
    frames = range(0, timePoints, stepSize)

    # Initialize plots
    def init():
        # Upper plot (Intensity over time)
        ax1.set_facecolor('black')
        ax1.set_xlim(0, time[-1])
        ax1.set_ylim(0, maxIntensity * 1.2)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel(f'Time', fontsize=10, color='white')
        ax1.set_ylabel('Intensity', fontsize=10, color='white')
        ax1.grid(False)
        # Lower plot (Ribosome movement)
        ax2.set_facecolor('black')
        ax2.set_xlim(0, geneLength + 1)
        ax2.set_ylim(0.09, 0.15)
        ax2.axis('off')
        ax2.grid(False)
        return []

    # Animation function
    def animate(frame_idx):
        tp = frame_idx
        ax1.clear()
        ax2.clear()
        # Plot settings for upper plot
        ax1.set_facecolor('black')
        ax1.set_xlim(0, time[-1])
        ax1.set_ylim(0, maxIntensity * 1.2)
        ax1.set_xlabel(f'Time', fontsize=10, color='white')
        ax1.set_ylabel('Intensity', fontsize=10, color='white')
        ax1.plot([0, time[-1]], [0, 0], '-w', linewidth=2)
        ax1.plot([0, 0], [0, maxIntensity * 1.1], '-w', linewidth=2)
        # add white axis ticks
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        # Plot intensity
        if IntensityVector[tp] > 0 :
            ax1.plot(time[tp], IntensityVector[tp], 'o', markersize=5,
                    markeredgecolor=color, markerfacecolor=color)
        ax1.plot(time[:tp], IntensityVector[:tp], '-', color=color, linewidth=2)

        if SecondIntensityVector is not None:
            if SecondIntensityVector[tp] > 0 :
                ax1.plot(time[tp], SecondIntensityVector[tp], 's', markersize=5,
                        markeredgecolor=second_color, markerfacecolor=second_color)
            ax1.plot(time[:tp], SecondIntensityVector[:tp], '-', color=second_color, linewidth=2)

        # Plot perturbation line and label
        if timePerturbationApplication is not None:
            if time[tp] >= timePerturbationApplication:
                ax1.text(5, maxIntensity * 1.3, 'Harringtonine', color='cyan', fontsize=12)
                ax1.plot([timePerturbationApplication, timePerturbationApplication],
                         [0, maxIntensity * 1.3],color='cyan', linewidth=2, linestyle='-')
        # Add title on the first frame
        ax1.text(time[-1] / 2.3, maxIntensity * 1.4, 'Ribosome Movement',
                color='white', fontsize=14)
        ax1.grid(False)
        # Plot settings for lower plot
        ax2.set_facecolor('black')
        ax2.set_xlim(0, geneLength + 1)
        ax2.set_ylim(0.0, 0.15)
        ax2.axis('off')
        ax2.set_xlabel('Gene length:' + str(geneLength-1) , fontsize=10, color='white')
        
        # Plot gene line and probes
        ax2.plot([0, geneLength], [0.1, 0.1], 'w-', linewidth=2)
        ax2.plot(probePositions, [0.1] * len(probePositions), 's',
                 markersize=3, markeredgecolor=color, markerfacecolor=color)
        if second_probePositions is not None:
            ax2.plot(second_probePositions,[0.1] * len(second_probePositions), 's',
                    markersize=4, markeredgecolor=second_color, markerfacecolor=second_color)
        # Plot ribosomes
        for i in range(Max_No_Ribosomes):
            #numberOfProbesPassed_Second = 0
            position = RibosomePositions[i, tp]
            if position > 0 and position <= geneLength:
                # Ribosome body
                ribosome_color = 'w' # [0.7, 0.7, 0.7]
                ax2.plot(position, 0.095, 'o', markersize=10,
                         markeredgecolor=ribosome_color,
                         markerfacecolor=ribosome_color)
                ax2.plot(position, 0.1, 'o', markersize=9,
                         markeredgecolor=ribosome_color,
                         markerfacecolor=ribosome_color)                
                # activity indicator for second probe
                if second_probePositions is not None:
                    numberOfProbesPassed_Second = np.sum(np.array(second_probePositions) < position) #int(np.sum(second_probePositions <= position) / len(second_probePositions))
                    markerSize =  3 
                    for j in range(numberOfProbesPassed_Second): 
                        if numberOfProbesPassed_Second>0 and j ==numberOfProbesPassed_Second-1:
                            probe_color = color
                        else:
                            probe_color = second_color
                        ax2.plot(position+j*2, 0.11 + j*0.007, 'o', markersize=markerSize,
                            markeredgecolor=probe_color, markerfacecolor=probe_color)
                # Ribosome activity indicator
                numberOfProbesPassed_First =  np.sum(np.array(probePositions) < position)#int( np.sum(probePositions <= position) / len(probePositions) )
                markerSize = 0.3 * numberOfProbesPassed_First
                if second_probePositions is not None and numberOfProbesPassed_Second > 0 and SecondIntensityVector is not None:
                    probe_color = second_color
                else:
                    probe_color = color
                ax2.plot(position, 0.102, 'o', markersize=markerSize,
                         markeredgecolor=probe_color, markerfacecolor=probe_color)
        # Time label
        time_str = f'{time[tp]:.0f} s'
        ax2.text(geneLength + 10, 0.1, time_str, color='white', fontsize=8)
        ax2.set_xlabel('Gene length:' + str(geneLength-1) , fontsize=10, color='white')
        return []
    ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False,
                        interval=500 / FrameVelocity)
    # Save animation as GIF
    writergif = PillowWriter(fps=FrameVelocity)
    ani.save(f'{fileNameGif}.gif', writer=writergif)
    display(IPImage(filename= f'{fileNameGif}.gif'   ))
    plt.close(fig)






def plot_spot(amplitude, sigma=2, grid_size=13):
    mu_x, mu_y = (grid_size/2)-0.5, (grid_size/2)-0.5
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    x, y = np.meshgrid(x, y)
    z = amplitude * np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
    # Normalize and return as uint8
    if z.max() > 0:
        z = (255*(z/z.max())).astype(np.uint8)
    else:
        z = z.astype(np.uint8)
    return z




def plot_RibosomeMovement_and_Microscope(
    RibosomePositions,
    IntensityVector,
    probePositions,
    frame_rate=1,
    SecondIntensityVector=None,
    second_probePositions=None,
    fileNameGif='temp_gif',
    color='red',
    second_color='lime',
    FrameVelocity=10,
    timePerturbationApplication=None,
    pause_location=None,
):

    def _tint_grayscale(z, color_like):
        rgb = np.array(to_rgb(color_like)).reshape(1, 1, 3)  # (1,1,3)
        zf = np.clip(z, 0, 255).astype(np.float32) / 255.0   # (H,W) in [0,1]
        return zf[..., None] * rgb                           # (H,W,3) float in [0,1]

    # --- time / sizes ---
    time = np.arange(0, len(IntensityVector), 1) * frame_rate
    geneLength = int(np.nanmax(RibosomePositions))
    Max_No_Ribosomes, num_timepoints = RibosomePositions.shape
    timePoints = len(time)

    # --- safety normalize ---
    def _safe_norm(v):
        vmax = np.nanmax(v) if np.nanmax(v) != 0 else 1.0
        return v / vmax

    IntensityVector = _safe_norm(np.asarray(IntensityVector))
    if SecondIntensityVector is not None:
        SecondIntensityVector = _safe_norm(np.asarray(SecondIntensityVector))
    maxIntensity = 1.0

    # --- figure layout (unchanged except size tweak) ---
    fig = plt.figure(figsize=(12, 5.8), facecolor='black')
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.0, 1.0],
        width_ratios=[3.0, 0.12, 1.4],
        wspace=0.25, hspace=0.20
    )
    ax1 = fig.add_subplot(gs[0, 0:2])  # Intensity
    ax2 = fig.add_subplot(gs[1, 0:2])  # Ribosome movement

    n_crops = 2 if SecondIntensityVector is not None else 1
    subgs = gridspec.GridSpecFromSubplotSpec(n_crops, 1, subplot_spec=gs[0, 2], hspace=0.25)
    ax3 = fig.add_subplot(subgs[0, 0])                          # Ch 0 crop (tinted with `color`)
    ax4 = fig.add_subplot(subgs[1, 0]) if n_crops == 2 else None  # Ch 1 crop (tinted with `second_color`)

    normalized_intensity_vector_first_signal = IntensityVector
    normalized_intensity_vector_second_signal = SecondIntensityVector if SecondIntensityVector is not None else None

    stepSize = 5
    frames = range(0, timePoints, stepSize)

    def _style_crop_axis(ax, title):
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_facecolor('black')
        ax.set_aspect('equal')

    def init():
        ax1.set_facecolor('black')
        ax1.set_xlim(0, time[-1] if len(time) else 1)
        ax1.set_ylim(0, maxIntensity * 1.2)
        ax1.set_xlabel('Time (s)', fontsize=10, color='white')
        ax1.set_ylabel('Intensity (a.u.)', fontsize=10, color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(False)

        ax2.set_facecolor('black')
        ax2.set_xlim(0, geneLength + 1)
        ax2.set_ylim(0.0, 0.15)
        ax2.set_xlabel(f'Gene length: {geneLength-1}', fontsize=10, color='white')
        ax2.axis('off')
        ax2.grid(False)

        _style_crop_axis(ax3, 'Crop · Ch 0')
        if ax4 is not None:
            _style_crop_axis(ax4, 'Crop · Ch 1')
        return []

    def animate(frame_idx):
        tp = frame_idx
        ax1.cla(); ax2.cla(); ax3.cla()
        if ax4 is not None: ax4.cla()

        # -------- Intensity (unchanged styling) --------
        ax1.set_facecolor('black')
        ax1.set_xlim(0, time[-1] if len(time) else 1)
        ax1.set_ylim(0, maxIntensity * 1.2)
        ax1.set_xlabel('Time (s)', fontsize=10, color='white')
        ax1.set_ylabel('Intensity (a.u.)', fontsize=10, color='white')
        ax1.tick_params(axis='x', colors='white'); ax1.tick_params(axis='y', colors='white')
        ax1.grid(False)
        if len(time) > 0: ax1.plot([0, time[-1]], [0, 0], '-w', linewidth=2)
        ax1.plot([0, 0], [0, maxIntensity * 1.1], '-w', linewidth=2)

        if IntensityVector[tp] > 0:
            ax1.plot(time[tp], IntensityVector[tp], 'o', markersize=5,
                     markeredgecolor=color, markerfacecolor=color)
        ax1.plot(time[:tp+1], IntensityVector[:tp+1], '-', color=color, linewidth=2, label='Ch 0')

        if SecondIntensityVector is not None:
            if SecondIntensityVector[tp] > 0:
                ax1.plot(time[tp], SecondIntensityVector[tp], 's', markersize=5,
                         markeredgecolor=second_color, markerfacecolor=second_color)
            ax1.plot(time[:tp+1], SecondIntensityVector[:tp+1], '-', color=second_color, linewidth=2, label='Ch 1')

        if timePerturbationApplication is not None and len(time) > 0 and time[tp] >= timePerturbationApplication:
            ax1.text(0.02 * time[-1], maxIntensity * 1.28, 'Harringtonine', color='cyan', fontsize=12)
            ax1.plot([timePerturbationApplication, timePerturbationApplication],
                     [0, maxIntensity * 1.28], color='cyan', linewidth=2, linestyle='-')
        ax1.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=9, loc='upper left')

        # -------- Ribosome movement (unchanged) --------
        ax2.set_facecolor('black')
        ax2.set_xlim(0, geneLength + 1)
        ax2.set_ylim(0.0, 0.15)
        ax2.set_xlabel(f'Gene length: {geneLength-1}', fontsize=10, color='white')
        ax2.axis('off'); ax2.grid(False)

        ax2.plot([0, geneLength], [0.1, 0.1], 'w-', linewidth=2)
        ax2.plot(probePositions, [0.1] * len(probePositions), 's',
                 markersize=3, markeredgecolor=color, markerfacecolor=color)
        if second_probePositions is not None:
            ax2.plot(second_probePositions, [0.1] * len(second_probePositions), 's',
                     markersize=4, markeredgecolor=second_color, markerfacecolor=second_color)

        for i in range(Max_No_Ribosomes):
            position = RibosomePositions[i, tp]
            if 0 < position <= geneLength:
                ax2.plot(position, 0.097, 'o', markersize=10, markeredgecolor='w', markerfacecolor='w')
                ax2.plot(position, 0.101, 'o', markersize=9, markeredgecolor='w', markerfacecolor='w')

                numberOfProbesPassed_Second = 0
                if second_probePositions is not None:
                    numberOfProbesPassed_Second = np.sum(np.array(second_probePositions) < position)
                    for j in range(numberOfProbesPassed_Second):
                        probe_color = color if (numberOfProbesPassed_Second > 0 and j == numberOfProbesPassed_Second - 1) else second_color
                        ax2.plot(position + j * 2, 0.105 + j * 0.0035, 'o', markersize=3,
                                 markeredgecolor=probe_color, markerfacecolor=probe_color)

                numberOfProbesPassed_First = np.sum(np.array(probePositions) < position)
                markerSize = 0.3 * numberOfProbesPassed_First
                probe_color = (second_color if (second_probePositions is not None and numberOfProbesPassed_Second > 0 and SecondIntensityVector is not None)
                               else color)
                ax2.plot(position, 0.102, 'o', markersize=markerSize,
                         markeredgecolor=probe_color, markerfacecolor=probe_color)

        time_str = f'{time[tp]:.0f} s' if len(time) else '0 s'
        ax2.text(geneLength + 10, 0.1, time_str, color='white', fontsize=8)
        # plot a vertical white line if pause_location is given
        if pause_location is not None:
            ax2.plot([pause_location, pause_location], [0.09, 0.15], color='white', linestyle='--', linewidth=1)

        # -------- Crops (now TINTED to channel colors) --------
        amplitude = normalized_intensity_vector_first_signal[tp]
        sigma = 1 + amplitude * 2
        z = plot_spot(amplitude, sigma=sigma)               # expects 2D array
        z_colored = _tint_grayscale(z, color)               # <- tint with `color`
        ax3.imshow(z_colored)                               # no cmap/vmax; already RGB
        _style_crop_axis(ax3, 'Crop · Ch 0')

        if ax4 is not None and normalized_intensity_vector_second_signal is not None:
            amplitude2 = normalized_intensity_vector_second_signal[tp]
            sigma2 = 1 + amplitude2 * 3
            z2 = plot_spot(amplitude2, sigma=sigma2)
            z2_colored = _tint_grayscale(z2, second_color)  # <- tint with `second_color`
            ax4.imshow(z2_colored)
            _style_crop_axis(ax4, 'Crop · Ch 1')

        return []

    ani = FuncAnimation(fig, animate, frames=frames, init_func=init, blit=False,
                        interval=1000 / FrameVelocity)

    if fileNameGif is None:
        plt.close(fig); return

    ani.save(f'{fileNameGif}.gif', writer=PillowWriter(fps=FrameVelocity))
    display(IPImage(filename=f'{fileNameGif}.gif'))
    plt.close(fig)


def plot_kymograph(list_occupancy_output,
                selected_trajectory = 0,
              figsize=(10,6),
              xlabel='Gene Position (codons)',
              ylabel='Time Steps',
              title='Kymograph',
              aspect='auto',
              interpolation='nearest',
              vmin=None,
              vmax=None,):
    rib_matrix_codon_time = list_occupancy_output[selected_trajectory].T # Transpose to have codons on x-axis and time on y-axis
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rib_matrix_codon_time, aspect=aspect, cmap='binary_r', interpolation=interpolation, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)
    plt.show()

def plot_ribosome_density(list_occupancy_output,
                data_sequence,
                selected_trajectory = 0,
                figsize=(12,3),
                smooth_window = 40,):
    rib_matrix_codon_time = list_occupancy_output[selected_trajectory].T # Transpose to have codons on x-axis and time on y-axis
    ribosome_occupancy = np.sum(rib_matrix_codon_time, axis=0)
    ribosome_occupancy /= np.sum(ribosome_occupancy)
    plt.figure(figsize=figsize)
    if smooth_window > 1:
        ribosome_occupancy_smooth = pd.Series(ribosome_occupancy).rolling(window=smooth_window, min_periods=1, center=True).mean().to_numpy()
        plt.bar(x=range(len(ribosome_occupancy_smooth)), height=ribosome_occupancy_smooth, width=5, color='lightgray', linewidth=0.5)
        plt.plot(ribosome_occupancy_smooth, color='k', linewidth=0.5)
    else:
        plt.bar(range(len(ribosome_occupancy)), ribosome_occupancy, color='k', alpha=0.5, width=1, label='Ribosome Occupancy', linewidth=0)
    # print a vertical line at the position of the pause site
        # check if pause_indexes exists in data_sequence AND is not None
    if 'pause_indexes' in data_sequence and data_sequence['pause_indexes'] is not None:
            plt.axvline(x=data_sequence['pause_indexes'], color='r', linestyle='-', linewidth=1)
    plt.xlim(0, data_sequence['gene_length'])
    plt.xlabel('Gene Position (codons)')
    plt.ylabel(r'$P\text{(ribosome at codon i)}$', fontsize=12, labelpad=8)
    plt.title('Ribosome Occupancy')
    plt.show()