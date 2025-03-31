# Data

The files here are only the event files for the parton and smeared events. They are numpy arrays contain the 4-momentum of the six jets and has the shape `[N, 6, 4]` where $N=12000$ is the number of events. There are $2000$ events per invariant mass bin.

The data used in `analysis.ipynb` is largely created by `postdata.py` which gets its data from running the algorithms themselves. Since those files take up $\mathcal{O}(10)$ GB, they are not available here but can be made available upon request.
