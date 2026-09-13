"""Tulya Semantic Frontier v2.

Exact finite semantic-partition benchmark with learned binary bottlenecks.
Designed for Kaggle and imported by kaggle_semantic_frontier.ipynb.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=7
N_BITS=12
OUT=Path("/kaggle/working/tulya_semantic_frontier")
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def all_states(d):
    z=np.arange(2**d,dtype=np.uint32)
    s=np.arange(d-1,-1,-1,dtype=np.uint32)
    return ((z[:,None]>>s[None,:])&1).astype(np.uint8)

X_np=all_states(N_BITS)
X=torch.tensor(X_np,dtype=torch.float32)

def pc(x): return x.sum(1).astype(np.int64)

T={}
for i in range(6):
    T[f"bit_{i}"]=lambda x,i=i:x[:,i].astype(np.uint8)
for a,b in [(0,1),(2,3),(0,5),(1,7),(3,9),(0,11)]:
    T[f"xor_{a}_{b}"]=lambda x,a=a,b=b:(x[:,a]^x[:,b]).astype(np.uint8)
T["parity_all"]=lambda x:(pc(x)%2).astype(np.uint8)
T["majority"]=lambda x:(pc(x)>x.shape[1]//2).astype(np.uint8)
T["popcount_mod3_zero"]=lambda x:(pc(x)%3==0).astype(np.uint8)
T["first_half_parity"]=lambda x:(x[:,:x.shape[1]//2].sum(1)%2).astype(np.uint8)
T["second_half_parity"]=lambda x:(x[:,x.shape[1]//2:].sum(1)%2).astype(np.uint8)

TASK_NAMES=list(T)
Y_np=np.stack([T[n](X_np) for n in TASK_NAMES],1)
Y=torch.tensor(Y_np,dtype=torch.float32)
NESTED=["bit_0","bit_1","xor_2_3","majority","first_half_parity","popcount_mod3_zero","parity_all"]
BASE=["bit_0","bit_1","xor_2_3"]

r=np.arange(1,len(TASK_NAMES)+1,dtype=float)
p=1/(r**1.15); p/=p.sum()
TASK_P=dict(zip(TASK_NAMES,p))

def as_rows(a):
    a=np.asarray(a)
    return a[:,None] if a.ndim==1 else a

def entropy_rows(a):
    _,c=np.unique(as_rows(a),axis=0,return_counts=True)
    p=c/c.sum()
    return float(-(p*np.log2(p)).sum())

def pstats(a):
    a=as_rows(a)
    K=len(np.unique(a,axis=0))
    H=entropy_rows(a)
    B=int(math.ceil(math.log2(K))) if K>1 else 0
    return K,H,B

def joint_rows(a,b): return np.concatenate([as_rows(a),as_rows(b)],1)

def conditional_entropy(a,given):
    return max(0.0,entropy_rows(joint_rows(a,given))-entropy_rows(given))

def mutual_information(a,b):
    return max(0.0,entropy_rows(a)+entropy_rows(b)-entropy_rows(joint_rows(a,b)))

def partition_metrics(s,z,tol=1e-12):
    hsz=conditional_entropy(s,z)
    hzs=conditional_entropy(z,s)
    return {
        "semantic_entropy_bits":entropy_rows(s),
        "code_entropy_bits":entropy_rows(z),
        "h_semantic_given_code":hsz,
        "h_code_given_semantic":hzs,
        "mutual_information_semantic_code":mutual_information(s,z),
        "semantic_sufficient":bool(hsz<=tol),
        "partition_equivalent":bool(hsz<=tol and hzs<=tol),
    }

def exact_frontier():
    rows=[]
    for k in range(1,len(NESTED)+1):
        ids=[TASK_NAMES.index(n) for n in NESTED[:k]]
        K,H,B=pstats(Y_np[:,ids])
        rows.append({"task_count":k,"tasks":",".join(NESTED[:k]),"classes":K,
                     "entropy_bits":H,"fixed_bits":B,"raw_bits":N_BITS})
    return pd.DataFrame(rows)

def surprise_novelty():
    ids=[TASK_NAMES.index(n) for n in BASE]
    _,H0,_=pstats(Y_np[:,ids])
    rows=[]
    for name in TASK_NAMES:
        if name in BASE: continue
        _,H1,_=pstats(Y_np[:,ids+[TASK_NAMES.index(name)]])
        rows.append({"task":name,"probability":TASK_P[name],
                     "task_surprise_bits":-math.log2(TASK_P[name]),
                     "semantic_novelty_bits":H1-H0})
    return pd.DataFrame(rows)

class Net(nn.Module):
    def __init__(self,bits,out,hidden=128):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(N_BITS,hidden),nn.ReLU(),
                               nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,bits))
        self.dec=nn.Sequential(nn.Linear(bits,hidden),nn.ReLU(),nn.Linear(hidden,out))
    def encode(self,x):
        prob=torch.sigmoid(self.enc(x))
        hard=(prob>=.5).float()
        return hard.detach()-prob.detach()+prob,prob
    def forward(self,x):
        z,p=self.encode(x)
        return self.dec(z),z,p

@torch.no_grad()
def all_codes(model):
    p=torch.sigmoid(model.enc(X.to(DEVICE)))
    return (p>=.5).to(torch.uint8).cpu().numpy()

def future_task_diagnostics(code,current_tasks):
    unseen=[t for t in TASK_NAMES if t not in set(current_tasks)]
    mass=sum(TASK_P[t] for t in unseen)
    rows=[]
    for name in unseen:
        y=Y_np[:,TASK_NAMES.index(name)]
        h=conditional_entropy(y,code)
        rows.append({"task":name,"probability":TASK_P[name],
                     "task_surprise_bits":-math.log2(TASK_P[name]),
                     "h_task_given_code_bits":h,"recoverable":bool(h<=1e-12)})
    df=pd.DataFrame(rows)
    if rows:
        expected=sum((TASK_P[r["task"]]/mass)*r["h_task_given_code_bits"] for r in rows)
        weighted=sum(TASK_P[r["task"]]*r["h_task_given_code_bits"] for r in rows)
    else:
        expected=weighted=0.0
    return df,float(expected),float(weighted),float(mass)

@dataclass
class Cfg:
    tasks:tuple
    bottleneck_bits:int
    n_train_states:int=1024
    steps:int=3000
    eval_every:int=100
    lr:float=2e-3
    wd:float=1e-4
    hidden:int=128
    seed:int=0

def run(cfg:Cfg):
    seed_all(cfg.seed)
    ids=[TASK_NAMES.index(n) for n in cfg.tasks]
    yy=Y[:,ids]
    semantic=Y_np[:,ids]
    rng=np.random.default_rng(cfg.seed)
    tri=rng.choice(len(X),min(cfg.n_train_states,len(X)),replace=False)
    tr=torch.tensor(tri,dtype=torch.long)

    m=Net(cfg.bottleneck_bits,len(ids),cfg.hidden).to(DEVICE)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    xt=X[tr].to(DEVICE); yt=yy[tr].to(DEVICE)
    xa=X.to(DEVICE); ya=yy.to(DEVICE)
    hist=[]; fut=[]

    for step in range(cfg.steps+1):
        if step:
            m.train()
            logits,_,_=m(xt)
            loss=F.binary_cross_entropy_with_logits(logits,yt)
            opt.zero_grad(); loss.backward(); opt.step()

        if step%cfg.eval_every==0 or step==cfg.steps:
            m.eval()
            with torch.no_grad():
                tp=(m(xt)[0]>=0).float()
                ap=(m(xa)[0]>=0).float()
                train_acc=float((tp==yt).float().mean())
                all_acc=float((ap==ya).float().mean())
                joint_train=float((tp==yt).all(1).float().mean())
                joint_all=float((ap==ya).all(1).float().mean())

            z=all_codes(m)
            pm=partition_metrics(semantic,z)
            fd,exp_unc,w_unc,mass=future_task_diagnostics(z,cfg.tasks)
            rec=fd.loc[fd.recoverable,"task"].tolist()
            hist.append({
                "step":step,"train_acc":train_acc,"all_state_acc":all_acc,
                "joint_train_accuracy":joint_train,"joint_all_state_accuracy":joint_all,
                "learned_classes":len(np.unique(z,axis=0)),
                **pm,
                "future_recoverable_count":len(rec),
                "future_recoverable_tasks":"|".join(rec),
                "expected_future_semantic_uncertainty_bits":exp_unc,
                "prior_weighted_future_unresolved_bits":w_unc,
                "future_task_probability_mass":mass,
            })
            for _,fr in fd.iterrows():
                fut.append({"step":step,**fr.to_dict()})

    hist=pd.DataFrame(hist); future_hist=pd.DataFrame(fut)
    z=all_codes(m)
    pm=partition_metrics(semantic,z)
    K0,H0,B0=pstats(semantic)
    fd,exp_unc,w_unc,mass=future_task_diagnostics(z,cfg.tasks)
    rec=fd.loc[fd.recoverable,"task"].tolist()
    last=hist.iloc[-1]
    summary={
        "tasks":list(cfg.tasks),"bottleneck_bits":cfg.bottleneck_bits,
        "n_train_states":cfg.n_train_states,"steps":cfg.steps,
        "theoretical_classes":K0,"theoretical_entropy_bits":H0,"theoretical_fixed_bits":B0,
        "learned_classes":len(np.unique(z,axis=0)),"learned_code_entropy_bits":pm["code_entropy_bits"],
        **{k:pm[k] for k in ["h_semantic_given_code","h_code_given_semantic",
                             "mutual_information_semantic_code","semantic_sufficient","partition_equivalent"]},
        "final_train_acc":float(last.train_acc),"final_all_state_acc":float(last.all_state_acc),
        "final_joint_train_accuracy":float(last.joint_train_accuracy),
        "final_joint_all_state_accuracy":float(last.joint_all_state_accuracy),
        "future_recoverable_tasks":rec,
        "expected_future_semantic_uncertainty_bits":exp_unc,
        "prior_weighted_future_unresolved_bits":w_unc,
        "future_task_probability_mass":mass,
    }
    return m,hist,future_hist,summary

def plot_demo(hist,summary):
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(hist.step,hist.train_acc,label="component train")
    ax.plot(hist.step,hist.all_state_acc,label="component all-state")
    ax.plot(hist.step,hist.joint_all_state_accuracy,label="joint all-state")
    ax.set(xlabel="step",ylabel="accuracy",ylim=(0,1.02),title="Memorization vs exhaustive generalization")
    ax.grid(alpha=.25); ax.legend(); plt.show()

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(hist.step,hist.h_semantic_given_code,label="H(S|Z): missing semantic bits")
    ax.plot(hist.step,hist.h_code_given_semantic,label="H(Z|S): excess retained bits")
    ax.set(xlabel="step",ylabel="bits",title="Exact partition diagnostics")
    ax.grid(alpha=.25); ax.legend(); plt.show()

    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(hist.step,hist.expected_future_semantic_uncertainty_bits,
            label="expected H(future task | Z)")
    ax.set(xlabel="step",ylabel="bits",title="Future-task semantic uncertainty")
    ax.grid(alpha=.25); ax.legend(); plt.show()

def run_all(run_sweep=False,run_long=False):
    OUT.mkdir(parents=True,exist_ok=True)
    seed_all()
    print("torch",torch.__version__,"device",DEVICE)
    if torch.cuda.is_available(): print("GPU",torch.cuda.get_device_name(0))

    frontier=exact_frontier()
    frontier.to_csv(OUT/"exact_frontier.csv",index=False)
    print("\nExact frontier"); print(frontier.to_string(index=False))

    novelty=surprise_novelty()
    novelty.to_csv(OUT/"task_surprise_vs_semantic_novelty.csv",index=False)

    Q=tuple(NESTED[:4])
    ids=[TASK_NAMES.index(n) for n in Q]
    _,_,B=pstats(Y_np[:,ids])
    print("\nExact fixed-bit bound for demo:",B)
    model,hist,future,summary=run(Cfg(Q,B,1024,3000,100,seed=SEED))
    hist.to_csv(OUT/"demo_trace.csv",index=False)
    future.to_csv(OUT/"demo_future_task_trace.csv",index=False)
    (OUT/"demo_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    plot_demo(hist,summary)

    if run_sweep:
        rows=[]
        for n in [256,512,1024,2048]:
            for bits in sorted(set([max(1,B-1),B,min(N_BITS,B+2)])):
                for steps in [500,1500,4000]:
                    _,_,_,s=run(Cfg(Q,bits,n,steps,max(100,steps//10),seed=0))
                    rows.append({
                        "n":n,"bits":bits,"steps":steps,"theory_bits":B,
                        "component_all_state_acc":s["final_all_state_acc"],
                        "joint_all_state_accuracy":s["final_joint_all_state_accuracy"],
                        "h_semantic_given_code":s["h_semantic_given_code"],
                        "h_code_given_semantic":s["h_code_given_semantic"],
                        "mutual_information_semantic_code":s["mutual_information_semantic_code"],
                        "semantic_sufficient":s["semantic_sufficient"],
                        "partition_equivalent":s["partition_equivalent"],
                        "future_recoverable_count":len(s["future_recoverable_tasks"]),
                        "future_recoverable_tasks":"|".join(s["future_recoverable_tasks"]),
                        "expected_future_semantic_uncertainty_bits":s["expected_future_semantic_uncertainty_bits"],
                        "prior_weighted_future_unresolved_bits":s["prior_weighted_future_unresolved_bits"],
                    })
        pd.DataFrame(rows).to_csv(OUT/"n_bits_compute_sweep.csv",index=False)

    if run_long:
        Qg=("parity_all","first_half_parity","second_half_parity")
        ids=[TASK_NAMES.index(n) for n in Qg]
        _,_,Bg=pstats(Y_np[:,ids])
        gm,gh,gf,gs=run(Cfg(Qg,max(Bg,5),768,30000,100,1e-3,1e-2,192,3))
        gh.to_csv(OUT/"grokking_trace.csv",index=False)
        gf.to_csv(OUT/"grokking_future_task_trace.csv",index=False)
        (OUT/"grokking_summary.json").write_text(json.dumps(gs,indent=2))

    metadata={
        "version":2,"seed":SEED,"n_bits":N_BITS,"n_states":len(X_np),
        "task_names":TASK_NAMES,"nested_tasks":NESTED,"base_tasks":BASE,
        "task_probabilities":TASK_P,
        "definitions":{
            "h_semantic_given_code":"H(S|Z): present-task semantic information missing from learned code",
            "h_code_given_semantic":"H(Z|S): distinctions retained beyond present-task semantic sufficiency",
            "partition_equivalent":"True iff H(S|Z)=0 and H(Z|S)=0 up to tolerance",
            "future_task_uncertainty":"H(Y_t|Z) for an unseen task t",
            "task_surprise":"-log2 P(t); distinct from semantic uncertainty",
        },
    }
    (OUT/"benchmark_metadata.json").write_text(json.dumps(metadata,indent=2))
    print("\nSaved:",sorted(p.name for p in OUT.iterdir()))
    return summary

if __name__=="__main__":
    run_all(False,False)
