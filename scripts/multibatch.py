import subprocess

import numpy as np

if __name__ == "__main__":
    alg = "maqaoa"
    etype = "ttbar"
    dtype = "parton"
    qc = "H0"
    p = 8
    invmlows = ["1", "1.25", "1.5", "1.75", "2", "2.5"]
    invmhis = ["1.25", "1.5", "1.75", "2", "2.5", "3"]

    alg_str = f"--algorithm {alg}"
    evt_str = f"--event {etype}"
    dat_str = f"--dtype {dtype}"
    qcf_str = f"--quadcoeff {qc}"
    dpt_str = f"--depth {p}"
    inv_str = f"--invmlow {' '.join(invmlows)} --invmhi {' '.join(invmhis)}"

    # nodes == 132, main == 84
    num_jobs = 84
    evts_per = 25
    # node01 == 0, node02 == 550, node03 == 1100, main == 1650
    offset = 1650
    for low_lim in np.arange(1, evts_per * (num_jobs // 6), evts_per):
        indlims = (str(offset + low_lim), str(offset + low_lim + 24))

        ind_str = f"--indlims {' '.join(indlims)}"

        cmd = f"python batch_run.py {alg_str} {evt_str} {dat_str} {qcf_str} {dpt_str} {inv_str} {ind_str}"
        cmd = cmd.split(" ")

        print(f"Running index range: {indlims[0]} -- {indlims[1]}")
        subprocess.run(cmd)
