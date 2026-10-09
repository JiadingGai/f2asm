"""Shared unit-explicit statistics; no benchmark-family dependencies."""
import math
import statistics


def summarize(samples):
    if not samples or any(not math.isfinite(x) or x<0 for x in samples): raise ValueError("Invalid samples")
    xs=sorted(samples)
    def percentile(q):
        pos=(len(xs)-1)*q; i=int(pos); w=pos-i
        return xs[i]*(1-w)+xs[min(i+1,len(xs)-1)]*w
    return {"count":len(xs),"minimum":xs[0],"median":statistics.median(xs),
            "mean":statistics.mean(xs),"p05":percentile(.05),"p95":percentile(.95),
            "maximum":xs[-1],"stdev":statistics.stdev(xs) if len(xs)>1 else 0}



def rates(workload,milliseconds):
    if not math.isfinite(milliseconds) or milliseconds<=0: raise ValueError("Positive event duration required")
    return {key+"_per_second":value*1000/milliseconds for key,value in workload.items()}
