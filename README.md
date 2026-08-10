# CINN_fmriprep

A lightweight Python package for submitting fMRIprep and MRIQC jobs on the racc2 HPC cluster. Handles SLURM job configuration, Singularity container setup, and job submission — so you only need to specify your paths and subject IDs.

Full walkthroughs are available on the [CINN Computational Lab documentation site](https://cinn-comp-lab.github.io).

---

## Installation

### With Pixi (recommended)

[Pixi](https://pixi.sh) is a fast, self-contained package manager that installs everything into the project directory — no module loads or conda activation needed.

**1. Install pixi** (one-time, run on racc2):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then start a new shell (or `source ~/.bashrc`) so the `pixi` command is available.

**2. Install the project environment:**

```bash
cd /path/to/CINN_fmriprep
pixi install
```

This reads `pixi.toml`, resolves all dependencies (including `simple-slurm`), and creates a `.pixi/` environment inside the project directory.

> **Home quota tip:** If your home directory is quota-limited, clone or move the project to `/storage/` before running `pixi install`.

**3. Point VSCode at the pixi Python interpreter:**

Open the project in VSCode (connected to racc2 via Remote-SSH), then find the path to the pixi environment's Python executable:

```bash
pixi run python -c "import sys; print(sys.executable)"
# e.g. /path/to/CINN_fmriprep/.pixi/envs/default/bin/python
```

Open any notebook in the `notebooks/` folder, click **Select Kernel** → **Python Environments** → **Enter interpreter path**, and paste the path printed above. VSCode will use this as the kernel for all notebooks in the project.

---

### With conda (alternative)

```bash
module load anaconda
conda create -n cinn_fmriprep python=3.10 pyyaml
conda activate cinn_fmriprep
pip install simple-slurm
pip install -e .
```

Then point VSCode at the conda environment's interpreter (usually `~/anaconda3/envs/cinn_fmriprep/bin/python`, or the `--prefix` path if you installed to storage).

---

## Quick start

All usage is via the Python API (typically from a Jupyter notebook). Example notebooks are in the `notebooks/` folder.

### fMRIprep — single subject

```python
from CINN_fmriprep import FmriPrepHandler

handler = FmriPrepHandler(
    bids_path  = '/storage/research/myproject/bids',
    out_path   = '/storage/research/myproject/derivatives',
    work_path  = '/storage/scratch/myproject/work',
    slurmout_path = '/storage/scratch/myproject/slurm_logs',
    subject    = 'sub-01',
)
handler.make_slurm()
handler.submit_slurm()
```

On submission, a message is printed showing the job name, expected output paths, HTML report location, and log file paths.

### fMRIprep — all subjects

```python
handler = FmriPrepHandler(..., subject='allsubs')
handler.make_slurm()
handler.submit_slurm()
```

### fMRIprep — multiple subjects (separate jobs)

```python
from CINN_fmriprep import MultipleFmriPrepHandler

handler = MultipleFmriPrepHandler(
    bids_path     = '/storage/research/myproject/bids',
    out_path      = '/storage/research/myproject/derivatives',
    work_path     = '/storage/scratch/myproject/work',
    slurmout_path = '/storage/scratch/myproject/slurm_logs',
    subjects      = ['sub-01', 'sub-02', 'sub-03'],
)
handler.make_fmriprep_handlers()
handler.make_slurms()
handler.submit_slurms()
```

### MRIQC

```python
from CINN_fmriprep import MriqcHandler

handler = MriqcHandler(
    bids_path     = '/storage/research/myproject/bids',
    out_path      = '/storage/research/myproject/derivatives/mriqc',
    work_path     = '/storage/scratch/myproject/work',
    slurmout_path = '/storage/scratch/myproject/slurm_logs',
    subject       = 'allsubs',   # or a specific subject ID
)
handler.make_slurm()
handler.submit_slurm()
```

---

## Configuration

SLURM resource defaults and Singularity paths are set in `config/config.yml`. You should not normally need to edit this unless the container images are updated.

Key settings:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `slurm` | `cpus_per_task` | 32 | CPUs per fMRIprep job |
| `slurm` | `mem_per_cpu` | 6G | Memory per CPU |
| `mriqc_slurm` | `cpus_per_task` | 8 | CPUs per MRIQC job |
| `paths` | `fs_license` | (pre-set) | FreeSurfer licence file |
| `paths` | `tf_path` | (pre-set) | TemplateFlow cache directory |

---

## Monitoring jobs

Use the **VSCode SLURM Dashboard** extension to monitor running jobs and view progress. The submission message printed after `submit_slurm()` shows the exact `.out` and `.err` log file paths.
