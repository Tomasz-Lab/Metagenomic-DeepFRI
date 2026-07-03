import os
import shutil
import subprocess
import time

import numpy as np

np.random.seed(42)


def generate_fasta(count, filename):
    """Randomly sampple sequences from SwissProt database and write to a FASTA file."""
    command = [
        'conda', 'run', '-n', 'data-analysis', 'seqkit', 'sample', '-n',
        str(count), '-s', '42', '-o', filename, '-2',
        '../data/databases/afdb_swissprot_v4.fasta.gz'
    ]
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,
                       universal_newlines=True)
    except subprocess.CalledProcessError as err:
        print(f"seqkit sample failed for count={count}: {err.stderr}")
        return False

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
    warmup_runs = max(0, int(os.environ.get('BENCH_WARMUP_RUNS', '1')))

    # Create temporary directory for outputs
    temp_dir = 'benchmark_temp'
    os.makedirs(temp_dir, exist_ok=True)

    results = []
    tsv_path = 'benchmark_results.tsv'

    def persist_results():
        header = [
            'count', 'eggnog_fast_time', 'eggnog_default_time',
            'eggnog_ultra-sensitive_time', 'deepfri_time'
        ]
        with open(tsv_path, 'w') as f:
            f.write("\t".join(header) + "\n")
            for cnt in counts:
                row = [str(cnt)]
                for sens in ['fast', 'default', 'ultra-sensitive']:
                    time_entry = next(
                        (r for r in results
                         if r['count'] == cnt and f'eggnog_{sens}_time' in r),
                        None)
                    row.append(f"{time_entry[f'eggnog_{sens}_time']:.2f}"
                               if time_entry else "NA")
                deepfri_entry = next(
                    (r for r in results
                     if r['count'] == cnt and 'deepfri_time' in r), None)
                row.append(f"{deepfri_entry['deepfri_time']:.2f}"
                           if deepfri_entry else "NA")
                f.write("\t".join(row) + "\n")

    fasta_files = []

    def run_step(cmd,
                 label,
                 result_key,
                 count_value,
                 log_path=None,
                 measure=True):
        print(f"Running: {' '.join(cmd)}")
        start_ts = time.time()
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.PIPE
        log_file = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_file = open(log_path, "a", encoding="utf-8")
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
        if measure:
            results.append({'count': count_value, result_key: duration})
            persist_results()
        return duration

    def run_with_warm_cache(cmd,
                            label,
                            result_key,
                            count_value,
                            log_path=None):
        # Warmup runs populate file-system and tool-level caches.
        for run_idx in range(warmup_runs):
            warm_label = f"{label} warmup {run_idx + 1}/{warmup_runs}"
            warm_log = (f"{log_path}.warmup{run_idx + 1}.log"
                        if log_path else None)
            warm_duration = run_step(cmd,
                                     warm_label,
                                     result_key,
                                     count_value,
                                     log_path=warm_log,
                                     measure=False)
            if warm_duration is None:
                return None

        return run_step(cmd,
                        label,
                        result_key,
                        count_value,
                        log_path=log_path,
                        measure=True)

    print("Generating FASTA files...")
    for count in counts:
        filename = os.path.join(temp_dir, f'protein_sequences_{count}.fasta')
        if generate_fasta(count, filename):
            fasta_files.append((count, filename))
            print(f"Generated {filename}")
        else:
            print(f"Skipping count={count} due to generation failure")
            persist_results()

    try:
        for count_value, fasta in fasta_files:
            abs_fasta = os.path.abspath(fasta)
            print(f"\nProcessing {fasta}...")

            # Benchmarking EggNOG
            sensitivities = {
                'fast': ['--sensmode', 'fast'],
                'default': [],
                'ultra-sensitive': ['--sensmode', 'ultra-sensitive']
            }

            for sens_name, sens_flags in sensitivities.items():
                eggnog_out_prefix = abs_fasta.replace('.fasta',
                                                      f'_eggnog_{sens_name}')

                # -m diamond for speed
                cmd = [
                    'conda', 'run', '-n', 'eggnog', 'emapper.py', '-i',
                    abs_fasta, '-o', eggnog_out_prefix, '--data_dir',
                    '../data/databases',
                    '--cpu', '16', '--override', '--pfam_realign', 'none'
                ] + sens_flags

                run_with_warm_cache(cmd, f"EggNOG {sens_name}",
                                    f'eggnog_{sens_name}_time', count_value)

            # Benchmarking Metagenomic-DeepFRI
            deepfri_out = abs_fasta.replace('.fasta', '')

            # Run Metagenomic-DeepFRI
            deepfri_log = os.path.join(temp_dir, f"deepfri_{count_value}.log")
            cmd = [
                'conda', 'run', '-n', 'deepfri', 'mDeepFRI',
                'predict-function', '-i', abs_fasta, '-d',
                '../data/databases/afdb_swissprot_v4',
                '-d',
                '../data/databases/highquality_clust30',
                '-w',
                '../onnx_deepfri_models/',
                '-o', deepfri_out, '-t', '16'
            ]

            run_with_warm_cache(cmd,
                                "Metagenomic-DeepFRI",
                                'deepfri_time',
                                count_value,
                                log_path=deepfri_log)
    finally:
        persist_results()


if __name__ == "__main__":
    run_benchmark()
