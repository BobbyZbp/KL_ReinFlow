"""Extract gradient norm statistics from wandb offline run files."""
import os
import glob
import json
import numpy as np

base = '/home/tw559/steering_pi/KL_ReinFlow/wandb/wandb'

target_runs = {
    'none': 'run-20260504_210428-7nzuzj0i',
    'perstep_exact': 'run-20260504_210428-dq0uo4j0',
    'perstep_hutchinson': 'run-20260504_210428-iu7fxa6o',
    'hutchinson': 'run-20260504_210359-nhi1tr7a',
    'exact': 'run-20260504_210359-fh0wpj1f',
}

from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb


def extract_grad_norms(wandb_file, max_records=500000):
    ds = DataStore()
    ds.open_for_scan(wandb_file)

    actor_gn = []
    critic_gn = []
    count = 0

    while count < max_records:
        try:
            data = ds.scan_data()
            if data is None:
                break
        except Exception:
            break
        count += 1

        record = pb.Record()
        try:
            record.ParseFromString(data)
        except Exception:
            continue

        if record.HasField('history'):
            for item in record.history.item:
                if item.key == 'actor/grad_norm':
                    try:
                        v = json.loads(item.value_json)
                        if v is not None and v == v:
                            actor_gn.append(float(v))
                    except Exception:
                        pass
                elif item.key == 'critic/grad_norm':
                    try:
                        v = json.loads(item.value_json)
                        if v is not None and v == v:
                            critic_gn.append(float(v))
                    except Exception:
                        pass

    ds.close()
    return np.array(actor_gn), np.array(critic_gn)


header = "{:<22} {:>6} | {:>9} {:>8} {:>9} | {:>10} {:>9} {:>10}".format(
    'Mode', 'N', 'Actor med', 'p95', 'max', 'Critic med', 'p95', 'max')
print(header)
print('-' * len(header))

for mode, run_id in target_runs.items():
    wandb_files = glob.glob(os.path.join(base, run_id, '*.wandb'))
    if not wandb_files:
        print("{:<22} NO FILE".format(mode))
        continue

    agn, cgn = extract_grad_norms(wandb_files[0])
    if len(agn) == 0:
        sz = os.path.getsize(wandb_files[0]) / 1024
        print("{:<22} {:>6} entries (file={:.0f}KB, critic_gn={})".format(
            mode, 0, sz, len(cgn)))
        continue

    line = "{:<22} {:>6} | {:>9.1f} {:>8.1f} {:>9.1f} | {:>10.1f} {:>9.1f} {:>10.1f}".format(
        mode, len(agn),
        np.median(agn), np.percentile(agn, 95), np.max(agn),
        np.median(cgn), np.percentile(cgn, 95), np.max(cgn))
    print(line)
