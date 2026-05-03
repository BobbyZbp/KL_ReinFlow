# Cornell Unicorn Cluster — Personal Cheatsheet

> Personal reference for running the **ReinFlow** project on Cornell's Unicorn HPC cluster.
> Audience: future Claude sessions / agents working on this repo.

---

## 1. Identity & Access

| Field | Value |
|---|---|
| NetID | `bz292` |
| Email | `bz292@cornell.edu` |
| Institution | Cornell University |
| Cluster name | Unicorn (CoECIS Research IT) |
| Official docs | https://it.coecis.cornell.edu/researchit/using-the-unicorn-cluster/ |
| Support email | `itcoecis-help@cornell.edu` |

---

## 2. Connection

- **SSH host:** `unicorn-login-01.coecis.cornell.edu`
- **VPN:** Cornell VPN (Cisco Secure Client → `cuvpn.cuvpn.cornell.edu`) required if off-campus.
- **Login:**
  ```bash
  ssh bz292@unicorn-login-01.coecis.cornell.edu
  ```
- **Optional key auth (one-time, from local Ubuntu):**
  ```bash
  ssh-keygen -t ed25519
  ssh-copy-id bz292@unicorn-login-01.coecis.cornell.edu
  ```

---

## 3. Storage Layout (all paths are on the cluster)

| Path | Purpose | Notes |
|---|---|---|
| `/home/bz292` | Personal home | Quota'd. Shared across **all** nodes (NFS). |
| `/scratch/bz292` | User scratch (large data) | User-managed, no quota. Use this for datasets. |
| `/share/DATASERVER/` | Shared lab data | |
| `/scratch/datasets` | Public datasets | |
| `/tmp` | Per-node temporary | Auto-cleared on job end. |

Check quota: `quota -s`

**Key fact:** `/home` and `/scratch` are shared filesystems → anything you create on one node (conda envs, code, results) is visible from every other node.

### Project-local data convention (REINFLOW)

**Everything related to the ReinFlow project lives under `/home/bz292/ReinFlow/`** — no top-level scattering in `$HOME`.

| Path | Purpose | Env var |
|---|---|---|
| `~/ReinFlow/`        | code (rsync'd from local) | `REINFLOW_DIR` |
| `~/ReinFlow/data/`   | normalization, offline datasets | `REINFLOW_DATA_DIR` |
| `~/ReinFlow/log/`    | training logs, checkpoints | `REINFLOW_LOG_DIR` |
| `~/ReinFlow/hf_cache/` | HuggingFace downloads (pretrained ckpts) | (passed via `local_dir=`) |

`~/.bashrc` exports the env vars to these paths. Subsequent rsyncs from local **must exclude** `data/`, `log/`, `hf_cache/`, `wandb/` so cluster-side data isn't clobbered.

---

## 4. Project Locations

| Side | Path |
|---|---|
| Local Ubuntu | `/home/bobby/ReinFlow` |
| Cluster | `/home/bz292/ReinFlow` |
| Cluster logs | `/home/bz292/ReinFlow/logs/` |

---

## 5. Login Node Rules

**`unicorn-login-01` is for light work only:**
- ✅ editing files, `git`, file transfer, `conda` env creation, submitting Slurm jobs
- ❌ training, heavy data processing, long compilation, anything multi-core for >a few minutes

**Real compute must run on a compute node** via `salloc` (interactive) or `sbatch` (batch).

---

## 6. File Transfer (run from **local Ubuntu**, not from the cluster)

### Push code to cluster
```bash
rsync -avz --progress \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'wandb' --exclude 'logs' --exclude 'log' \
  --exclude 'data' --exclude 'hf_cache' --exclude 'ogbench' \
  /home/bobby/ReinFlow/ \
  bz292@unicorn-login-01.coecis.cornell.edu:/home/bz292/ReinFlow/
```

### Pull results back
```bash
rsync -avz \
  bz292@unicorn-login-01.coecis.cornell.edu:/home/bz292/ReinFlow/log/ \
  /home/bobby/ReinFlow/log/
```

### Tip
For large datasets (10+ GB), put them in `/scratch/bz292/` not `/home/bz292/`.

---

## 7. Slurm Gotchas (READ BEFORE USING `salloc`/`sbatch`)

- **Partition name uses underscore:** `default_partition` ✅ — NOT `default-partition` ❌
- Interactive `salloc` auto-routes to `default_partition-interactive` if no `-p` given.
- **Easiest:** omit `-p` entirely; the default works.
- If `salloc` errors with `getcwd failed: No such file or directory`, run `cd ~` first.

### Partitions available
| Partition | Priority | Use |
|---|---|---|
| `default_partition` | low | CPU/GPU jobs, default |
| `gpu` | medium | GPU-required jobs (preempts low) |
| group-specific | high | Faculty-owned servers (preempts lower) |

List partitions: `sinfo -s`

---

## 8. Conda Environment (ONE-TIME setup)

Do this on a **CPU compute node** (no GPU needed for installs):

```bash
# On login node
salloc --cpus-per-task=4 --mem=16G -t 1:00:00

# After allocation lands you on a compute node:
/share/apps/software/anaconda3/bin/conda init
exec bash                           # reload shell so (base) appears
cd ~/ReinFlow
conda create -n reinflow python=3.10 -y
conda activate reinflow
pip install -r requirements.txt     # adapt to ReinFlow's actual install command
exit                                 # release the node
```

The env lives at `/home/bz292/.conda/envs/reinflow/` and is usable from **any** node afterwards. `conda init` only needs to be run **once** ever (it edits `~/.bashrc`, which is shared).

---

## 9. Interactive GPU Session (debugging / first runs)

```bash
# On login node, optionally inside tmux so disconnect doesn't kill job
tmux new -s train

salloc --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 2:00:00
# wait, prompt becomes the compute node name

conda activate reinflow
cd ~/ReinFlow
nvidia-smi                          # confirm GPU
python <train_entry>.py --config <cfg>
exit                                 # releases GPU
```

`tmux` shortcuts: detach `Ctrl-b d`, reattach `tmux attach -t train`, list `tmux ls`.

### Available GPU types (request via `--gres=gpu:<type>:1`)
- `nvidia_rtx_6000_ada_generation` (seen in cluster docs)
- Generic: `--gres=gpu:1` lets the scheduler pick.

CUDA versions on cluster: 12.8.1 and 12.0 (default).

---

## 10. Batch GPU Job (long training, recommended)

Create `~/ReinFlow/run.sub`:

```bash
#!/bin/bash
#SBATCH -J reinflow
#SBATCH -o logs/reinflow_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate reinflow
cd $HOME/ReinFlow
python <train_entry>.py --config <cfg>
```

Submit and monitor:
```bash
mkdir -p ~/ReinFlow/logs
sbatch ~/ReinFlow/run.sub                                    # returns job id
squeue --me                                                  # PD=pending, R=running
tail -f ~/ReinFlow/logs/reinflow_<jobid>.out                 # live log
scancel <jobid>                                              # cancel
```

After `sbatch`, you can `exit` ssh — the job runs on its own.

---

## 11. Useful Commands Cheatsheet

| Command | Purpose |
|---|---|
| `squeue --me` | My queued/running jobs |
| `scancel <jobid>` | Cancel a job |
| `sinfo -s` | List partitions |
| `sinfo -o "%30N %10c %15m %30G"` | Node + GPU layout |
| `quota -s` | Disk usage on /home |
| `nvidia-smi` | GPU status (compute nodes only) |
| `salloc ...` | Get interactive compute node |
| `sbatch script.sub` | Submit batch job |
| `tmux new -s NAME` / `tmux attach -t NAME` | Persistent session |

---

## 12. Common Errors → Fixes

| Error | Fix |
|---|---|
| `getcwd failed: No such file or directory` | `cd ~` then retry |
| `invalid partition specified: default-partition-interactive` | Use `default_partition` (underscore) or omit `-p` |
| `salloc: error: Job submit/allocate failed: Invalid partition name specified` | Check partition name with `sinfo -s` |
| Job killed unexpectedly | Hit walltime (`-t`), memory limit (`--mem`), or login-node hard kill (don't run on login node) |

---

## 13. Workflow Summary (Quick Reference)

```
1. Local Ubuntu          → rsync code to cluster
2. ssh into login node   → light work only
3. salloc CPU node       → install conda env (one-time)
4. salloc GPU node OR sbatch → run training on compute node
5. rsync results back to local
```
