#!/usr/bin/env python3
from __future__ import annotations
import json, os, queue, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import pandas as pd
import torch

from experiment2_core import (
    DOMAINS, HARMFUL, RunSpec, prepare_datasets
)

V3_SEEDS=(100,101,102,103,104,105)
V3_ALLOWED={"no_event","stagnation","overfit_or_memorization"}

@dataclass(frozen=True)
class Recipe:
    domain:str
    intent:str
    expected_event:str
    name:str

RECIPES=(
    Recipe("modular_transformer","healthy","no_event","no_event"),
    Recipe("fashion_mnist_mlp","healthy","no_event","no_event"),
    Recipe("cifar10_cnn","healthy","no_event","no_event"),
    Recipe("synthetic_sequence_gru","high_lr","no_event","no_event"),

    Recipe("fashion_mnist_mlp","low_lr","stagnation","stagnation"),
    Recipe("cifar10_cnn","low_lr","stagnation","stagnation"),
    Recipe("synthetic_sequence_gru","low_lr","stagnation","stagnation"),

    Recipe("modular_transformer","small_train","overfit_or_memorization","overfit_or_memorization"),
    Recipe("cifar10_cnn","small_train","overfit_or_memorization","overfit_or_memorization"),
    Recipe("synthetic_sequence_gru","small_train","overfit_or_memorization","overfit_or_memorization"),
)

def specs(seeds:Sequence[int]=V3_SEEDS):
    ans=[]
    meta={}
    for r in RECIPES:
        for seed in seeds:
            sp=RunSpec(r.domain,r.intent,int(seed))
            ans.append(sp)
            meta[sp.run_id]=r
    return ans,meta

def _run_subprocess(gpu_id,sp,outroot,dataroot):
    p=Path(outroot)/sp.run_id/"summary.json"
    if p.exists():
        sm=json.loads(p.read_text())
        print(f"[GPU {gpu_id}] reuse {sp.run_id}: {sm['event']}",flush=True)
        return sm
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
    env["PYTHONUNBUFFERED"]="1"
    script=str((Path(__file__).parent/"experiment2_core.py").resolve())
    cmd=[sys.executable,script,"--worker",
         "--domain",sp.domain,"--intent",sp.intent,"--seed",str(sp.seed),
         "--outroot",str(outroot),"--dataroot",str(dataroot)]
    print(f"[GPU {gpu_id}] start {sp.run_id}",flush=True)
    cp=subprocess.run(cmd,env=env,text=True,capture_output=True)
    if cp.returncode!=0:
        raise RuntimeError(
            f"v3 worker failed on GPU {gpu_id}: {sp.run_id}\n"
            f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    sm=json.loads(p.read_text())
    print(
        f"[GPU {gpu_id}] done {sp.run_id}: {sm['event']} "
        f"observed={sm['event_observed']} @ {sm['event_time_fraction']:.3f}, "
        f"{sm['wall_time_sec']:.1f}s",flush=True
    )
    return sm

def _worker(gpu_id,q,outroot,dataroot):
    out=[]
    while True:
        try: sp=q.get_nowait()
        except queue.Empty: break
        try: out.append(_run_subprocess(gpu_id,sp,outroot,dataroot))
        finally: q.task_done()
    return out

def run_v3_suite(outroot,seeds:Sequence[int]=V3_SEEDS,
                 gpu_ids:Sequence[int]|None=None,
                 dataroot="/kaggle/working/tulya_data"):
    ss,meta=specs(seeds)
    root=Path(outroot);root.mkdir(parents=True,exist_ok=True)
    ids=list(gpu_ids if gpu_ids is not None else range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("v3 requires at least one visible CUDA GPU")

    prepare_datasets(dataroot)
    q=queue.Queue()
    for sp in ss:q.put(sp)

    print(f"Launching {len(ss)} v3 runs across {len(ids)} dynamically balanced GPU workers: {ids}",flush=True)
    with ThreadPoolExecutor(max_workers=len(ids)) as ex:
        futs=[ex.submit(_worker,g,q,outroot,dataroot) for g in ids]
        for fut in as_completed(futs):
            fut.result()

    rows=[]
    for sp in ss:
        p=root/sp.run_id/"summary.json"
        if not p.exists():raise RuntimeError(f"missing summary {p}")
        sm=json.loads(p.read_text())
        r=meta[sp.run_id]
        sm["recipe_name"]=r.name
        sm["recipe_expected_event"]=r.expected_event
        rows.append(sm)
    df=pd.DataFrame(rows)
    df.to_csv(root/"manifest.csv",index=False)
    return df

def validate_v3_manifest(manifest:pd.DataFrame):
    problems=[]

    bad=sorted(set(manifest.event)-V3_ALLOWED)
    if bad:problems.append(f"unexpected outcome classes: {bad}")

    cell=(manifest.groupby(["domain","recipe_name","recipe_expected_event"])
          .apply(lambda g:int((g.event==g.recipe_expected_event.iloc[0]).sum()),
                 include_groups=False)
          .rename("matches").reset_index())
    for _,r in cell.iterrows():
        if int(r.matches)<4:
            problems.append(
                f"{r.domain}/{r.recipe_name}: expected {r.recipe_expected_event}, "
                f"only {int(r.matches)}/6 matched"
            )

    for held in DOMAINS:
        te=manifest[manifest.domain==held]
        tr=manifest[manifest.domain!=held]
        if te.event.nunique()<2:
            problems.append(f"held-out {held}: fewer than 2 outcome classes")
        for event in sorted(te.event.unique()):
            n=int((tr.event==event).sum())
            if n<4:
                problems.append(
                    f"held-out {held}: class {event!r} has only {n} training-domain runs"
                )

    return problems,cell
