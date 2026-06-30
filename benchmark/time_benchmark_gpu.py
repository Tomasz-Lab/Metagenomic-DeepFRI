import gzip
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

np.random.seed(42)


def cleanup_file(file_path):
    """Delete a file if it exists."""
    if file_path.exists():
        print(f"[CLEANUP] Removing {file_path}")
        file_path.unlink()


def move_and_concat_prediction_files(source_dir: Path, tool_name: str,
                                     output_folder: Path):
    """
    Concatenate all gzipped prediction files in source_dir and move as a single gzipped file to output_folder.
    """
    pred_files = list(source_dir.glob("*preds*.gz"))
    if not pred_files:
        print(f"[WARN] No gzipped prediction files found in {source_dir}")
        return

    combined_file = output_folder / f"{tool_name}_go_preds.tsv"
    print(
        f"[CONCAT] Combining {len(pred_files)} gzipped files into {combined_file}"
    )

    with open(combined_file, "w") as outfile:
        for pred_file in pred_files:
            with gzip.open(pred_file, "rt") as infile:
                for line in infile:
                    outfile.write(line)
            # Remove the original file after concatenation
            pred_file.unlink()

    print(f"[DONE] Combined gzipped predictions saved to: {combined_file}")


def generate_fasta(count, filename):
    """Randomly sampple sequences from SwissProt database and write to a FASTA file."""
    command = [
        'conda', 'run', '-n', 'data-analysis', 'seqkit', 'sample', '-n',
        str(count), '-s', '42', '-o', filename, '-2',
        '../data/databases/afdb_swissprot_v4.fasta.gz'
    ]

    subprocess.run(command,
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE,
                   universal_newlines=True)

    # mutate 5-80% of sequences to make them novel
    mutated_filename = filename.replace('.fasta', '_mutated.fasta')
    with open(filename, 'r') as infile, open(mutated_filename, 'w') as outfile:
        for line in infile:
            if line.startswith('>'):
                outfile.write(line)
            else:
                seq = line.strip()
                seq_list = list(seq)
                num_mutations = np.random.randint(int(0.05 * len(seq)),
                                                  int(0.8 * len(seq)) + 1)
                mutation_indices = np.random.choice(len(seq_list),
                                                    size=num_mutations,
                                                    replace=False)
                for idx in mutation_indices:
                    original_aa = seq_list[idx]
                    possible_aas = [
                        aa for aa in 'ACDEFGHIKLMNPQRSTVWY'
                        if aa != original_aa
                    ]
                    seq_list[idx] = np.random.choice(possible_aas)
                mutated_seq = ''.join(seq_list)
                outfile.write(mutated_seq + '\n')
    # replace original file with mutated one
    shutil.move(mutated_filename, filename)
    return True


def run_benchmark():
    # generate FASTA file with 10, 100, 1000 and 10,000 sequences
    counts = [10, 100, 1000, 10000, 100_000]

    # Create temporary directory for outputs
    temp_dir = Path('benchmark_temp_gpu')
    temp_dir.mkdir(exist_ok=True)

    results = []
    tsv_path = 'benchmark_results_gpu.tsv'

    def persist_results():
        header = ['count', 'deepfri_time', 'deepgometa_time']
        with open(tsv_path, 'w') as f:
            f.write("\t".join(header) + "\n")
            for cnt in counts:
                row = [str(cnt)]

                deepfri_entry = next(
                    (r for r in results
                     if r['count'] == cnt and 'deepfri_time' in r), None)
                row.append(f"{deepfri_entry['deepfri_time']:.2f}"
                           if deepfri_entry else "NA")

                deepgometa_entry = next(
                    (r for r in results
                     if r['count'] == cnt and 'deepgometa_time' in r), None)
                row.append(f"{deepgometa_entry['deepgometa_time']:.2f}"
                           if deepgometa_entry else "NA")

                f.write("\t".join(row) + "\n")

    fasta_files = []

    def run_step(cmd, label, result_key, count_value, log_path=None):
        print(f"Running: {' '.join(cmd)}")
        start_ts = time.time()
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.PIPE
        log_file = None
        if log_path:
            # os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # log_path is Path or str
            path_obj = Path(log_path)
            path_obj.parent.mkdir(exist_ok=True, parents=True)
            log_file = open(path_obj, "a", encoding="utf-8")
            stdout_target = log_file
            stderr_target = subprocess.STDOUT
        try:
            subprocess.run(cmd,
                           check=True,
                           stdout=stdout_target,
                           stderr=stderr_target,
                           universal_newlines=True)
        except Exception as err:
            # Keep going but record the failure
            if isinstance(err, subprocess.CalledProcessError):
                print(
                    f"{label} failed with code {err.returncode}: {err.stderr}")
            else:
                print(f"{label} failed: {err}")
            if log_file:
                log_file.write(f"\n{label} failed: {err}\n")
                log_file.flush()
                log_file.close()
            persist_results()
            return None
        finally:
            if log_file and not log_file.closed:
                log_file.flush()
                log_file.close()
        duration = time.time() - start_ts
        print(f"{label} time: {duration:.2f} s")
        results.append({'count': count_value, result_key: duration})
        persist_results()
        return duration

    print("Generating FASTA files...")
    for count in counts:
        filename = temp_dir / f'protein_sequences_{count}.fasta'
        # generate_fasta expects str
        if generate_fasta(count, str(filename)):
            fasta_files.append((count, filename))
            print(f"Generated {filename}")
        else:
            print(f"Skipping count={count} due to generation failure")
            persist_results()

    try:
        for count_value, fasta_path in fasta_files:
            abs_fasta = str(fasta_path.resolve())
            print(f"\nProcessing {fasta_path}...")

            # Benchmarking Metagenomic-DeepFRI
            deepfri_out = str(fasta_path).replace('.fasta', '')

            # Run Metagenomic-DeepFRI
            deepfri_log = temp_dir / f"deepfri_{count_value}.log"
            cmd = [
                'conda', 'run', '-n', 'deepfri', 'mDeepFRI',
                'predict-function', '-i', abs_fasta, '-d',
                '../data/databases/afdb_swissprot_v4',
                '-d',
                '../data/databases/highquality_clust30',
                '-w',
                '../onnx_deepfri_models/',
                '-o', deepfri_out, '-t', '8'
            ]

            run_step(cmd,
                     "Metagenomic-DeepFRI",
                     'deepfri_time',
                     count_value,
                     log_path=str(deepfri_log))

            # Benchmarking DeepGOMeta
            script = "../protfunc_eval/vendor/deepgometa/predict.py"
            model = "../protfunc_eval/vendor/deepgometa/data"
            deepgo_log = temp_dir / f"deepgo_{count_value}.log"

            cmd_deepgo = [
                "conda", "run", "-n", "deepgometa", "python3", script,
                "--data-root", model, "-if", abs_fasta
            ]

            # Using run_step for deepgo
            if run_step(cmd_deepgo,
                        "DeepGOMeta",
                        'deepgometa_time',
                        count_value,
                        log_path=str(deepgo_log)):
                # Cleanup intermediate ESM file
                cleanup_file(fasta_path.parent / "example_esm.pkl")

                # Move predictions to output folder
                move_and_concat_prediction_files(fasta_path.parent,
                                                 f"deepgometa_{count_value}",
                                                 temp_dir)

    finally:
        persist_results()


if __name__ == "__main__":
    run_benchmark()
