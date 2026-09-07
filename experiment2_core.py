#!/usr/bin/env python3
from __future__ import annotations
import json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from kaggle_grokking_experiment import (
    CountSketchProjector, ExperimentConfig as GrokConfig, ModularTransformer,
    cosine, make_modular_dataset, spectral_stats,
)

DOMAINS=("modular_transformer","fashion_mnist_mlp","cifar10_cnn","synthetic_sequence_gru")
INTENTS=("healthy","high_lr","low_lr","small_train","strong_regularization")
HARMFUL={"divergence","stagnation","overfit_or_memorization"}

@dataclass(frozen=True)
class RunSpec:
    domain:str; intent:str; seed:int
    @property
    def run_id(self): return f"{self.domain}__{self.intent}__seed{self.seed}"

class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.a=nn.Linear(784,256); self.b=nn.Linear(256,128); self.o=nn.Linear(128,10)
    def forward(self,x,rep=False):
        x=F.relu(self.a(x.flatten(1))); h=F.relu(self.b(x)); y=self.o(h); return (y,h) if rep else y

class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.a=nn.Conv2d(3,32,3,padding=1); self.b=nn.Conv2d(32,64,3,padding=1); self.f=nn.Linear(64*8*8,128); self.o=nn.Linear(128,10)
    def forward(self,x,rep=False):
        x=F.max_pool2d(F.relu(self.a(x)),2); x=F.max_pool2d(F.relu(self.b(x)),2); h=F.relu(self.f(x.flatten(1))); y=self.o(h); return (y,h) if rep else y

class GRU(nn.Module):
    def __init__(self):
        super().__init__(); self.e=nn.Embedding(16,48); self.g=nn.GRU(48,96,batch_first=True); self.o=nn.Linear(96,4)
    def forward(self,x,rep=False):
        _,h=self.g(self.e(x)); h=h[-1]; y=self.o(h); return (y,h) if rep else y

class Noisy(Dataset):
    def __init__(self,base,seed,nc=10,p=.15):
        self.base=base; r=np.random.default_rng(seed); self.flip=r.random(len(base))<p; self.lab=r.integers(0,nc,len(base))
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]; return x,int(self.lab[i]) if self.flip[i] else y

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def dev(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def flat(m): return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def grads(m): return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in m.parameters()])
def l2(x): return float(torch.linalg.vector_norm(x.float()).item())
def chance(d): return .25 if d=="synthetic_sequence_gru" else (1/113 if d=="modular_transformer" else .1)
def progress(a,d): return float(np.clip((a-chance(d))/(1-chance(d)),0,1))
def specs(seeds=(0,1,2,3)): return [RunSpec(d,i,s) for d in DOMAINS for i in INTENTS for s in seeds]

def subset(ds,n,seed):
    if n<=0 or n>=len(ds): return ds
    ids=np.random.default_rng(seed).permutation(len(ds))[:n]; return Subset(ds,ids.tolist())

def image_setup(sp,root):
    if sp.domain=="fashion_mnist_mlp":
        tr=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.286,),(.353,))])
        a=datasets.FashionMNIST(root,train=True,download=True,transform=tr); v=datasets.FashionMNIST(root,train=False,download=True,transform=tr)
        m=MLP(); steps,lr,wd,n=1800,1e-3,1e-4,12000
    else:
        tr=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
        a=datasets.CIFAR10(root,train=True,download=True,transform=tr); v=datasets.CIFAR10(root,train=False,download=True,transform=tr)
        m=CNN(); steps,lr,wd,n=2200,1e-3,1e-4,12000
    if sp.intent=="high_lr": lr=.08 if sp.domain=="fashion_mnist_mlp" else .03
    if sp.intent=="low_lr": lr=2e-6
    if sp.intent=="small_train": n=500; steps=2600
    if sp.intent=="strong_regularization": wd=2.; lr=3e-4
    a=subset(a,n,sp.seed+11); v=subset(v,4000,sp.seed+19)
    if sp.intent=="small_train": a=Noisy(a,sp.seed+31)
    g=torch.Generator().manual_seed(sp.seed+101)
    return DataLoader(a,batch_size=256,shuffle=True,generator=g,num_workers=2,pin_memory=True),DataLoader(v,batch_size=512,num_workers=2,pin_memory=True),m,steps,lr,wd

def seq_setup(sp):
    r=np.random.default_rng(sp.seed+7); nt,nv,L=10000,3000,32
    x=r.integers(0,16,(nt,L)); xv=r.integers(0,16,(nv,L)); y=((x[:,0]+2*x[:,-1])%4); yv=((xv[:,0]+2*xv[:,-1])%4)
    steps,lr,wd,n=1600,1e-3,1e-4,nt
    if sp.intent=="high_lr": lr=.08
    if sp.intent=="low_lr": lr=1e-6
    if sp.intent=="small_train": n=400; steps=2200
    if sp.intent=="strong_regularization": wd=2.; lr=3e-4
    xt=torch.tensor(x[:n],dtype=torch.long); yt=torch.tensor(y[:n],dtype=torch.long)
    if sp.intent=="small_train":
        q=np.random.default_rng(sp.seed+44); mask=q.random(n)<.15; yt[torch.tensor(mask)]=torch.tensor(q.integers(0,4,mask.sum()))
    g=torch.Generator().manual_seed(sp.seed+101)
    return DataLoader(TensorDataset(xt,yt),256,shuffle=True,generator=g),DataLoader(TensorDataset(torch.tensor(xv),torch.tensor(yv)),512),GRU(),steps,lr,wd

@torch.no_grad()
def evaluate(m,loader,d):
    m.eval(); n=cor=0; loss=0.
    for x,y in loader:
        x,y=x.to(d,non_blocking=True),y.to(d,non_blocking=True); z=m(x); loss+=float(F.cross_entropy(z,y,reduction="sum")); cor+=int((z.argmax(1)==y).sum()); n+=len(y)
    return loss/n,cor/n

@torch.no_grad()
def probe(m,x):
    m.eval(); z,h=m(x,rep=True); p=F.softmax(z,-1); ent=-(p*p.clamp_min(1e-12).log()).sum(-1)
    s,_=spectral_stats(h,center=True,keep=0); std=float(h.float().std(unbiased=False)); ab=float(h.float().abs().mean())
    return dict(
        feature_activation_mean=float(h.float().mean()),feature_activation_std=std,feature_activation_abs_mean=ab,
        feature_activation_scale_ratio=std/max(ab,1e-12),
        feature_activation_saturation=float((h.float().abs()>3*max(std,1e-6)).float().mean()),
        feature_prediction_entropy_norm=float((ent/math.log(z.shape[-1])).mean()),feature_prediction_confidence=float(p.max(-1).values.mean()),
        feature_repr_effective_rank_norm=s["effective_rank_norm"],feature_repr_spectral_entropy_norm=s["spectral_entropy_norm"],
        feature_repr_participation_ratio_norm=s["participation_ratio_norm"],feature_repr_top1_energy=s["top1_energy"])

@torch.no_grad()
def probe_mod(m,x):
    m.eval(); z,dg=m(x,diagnostics=True); h=dg["final_representation"]; p=F.softmax(z,-1); ent=-(p*p.clamp_min(1e-12).log()).sum(-1)
    s,_=spectral_stats(h,center=True,keep=0); std=float(h.float().std(unbiased=False)); ab=float(h.float().abs().mean())
    return dict(feature_activation_mean=float(h.float().mean()),feature_activation_std=std,feature_activation_abs_mean=ab,
        feature_activation_scale_ratio=std/max(ab,1e-12),feature_activation_saturation=float((h.float().abs()>3*max(std,1e-6)).float().mean()),
        feature_prediction_entropy_norm=float((ent/math.log(z.shape[-1])).mean()),feature_prediction_confidence=float(p.max(-1).values.mean()),
        feature_repr_effective_rank_norm=s["effective_rank_norm"],feature_repr_spectral_entropy_norm=s["spectral_entropy_norm"],
        feature_repr_participation_ratio_norm=s["participation_ratio_norm"],feature_repr_top1_energy=s["top1_energy"])

def finish_features(df,domain):
    df["feature_train_progress"]=df.feature_train_acc.map(lambda x:progress(float(x),domain))
    first=max(float(df.iloc[0].feature_train_loss),1e-12); df["feature_train_loss_log_relative"]=np.log(df.feature_train_loss.clip(lower=1e-12)/first)
    return df

def label_event(df,domain):
    d=df.sort_values("meta_step_fraction").copy(); d["tp"]=d.feature_train_acc.map(lambda x:progress(x,domain)); d["vp"]=d.label_val_acc.map(lambda x:progress(x,domain))
    ini=max(float(d.iloc[0].feature_train_loss),1e-6)
    for _,r in d.iterrows():
        if not math.isfinite(r.feature_train_loss) or (r.meta_step_fraction>=.1 and r.feature_train_loss>max(10.,6*ini)): return "divergence",float(r.meta_step_fraction)
    early=d[d.meta_step_fraction<=.4]; late=d[d.meta_step_fraction>=.5]
    if len(early) and len(late) and (early.tp>=.9).any() and early.iloc[-1].vp<.45:
        q=late[late.vp>=.75]
        if len(q): return "delayed_improvement",float(q.iloc[0].meta_step_fraction)
    q=d[(d.meta_step_fraction<=.4)&(d.vp>=.65)]
    if len(q) and d.iloc[-1].vp>=.6: return "healthy_convergence",float(q.iloc[0].meta_step_fraction)
    q=d[(d.meta_step_fraction>=.4)&(d.tp>=.9)&(d.vp<.45)]
    if len(q): return "overfit_or_memorization",float(q.iloc[0].meta_step_fraction)
    q=d[d.meta_step_fraction>=.6]
    if len(q) and q.iloc[0].tp<.55 and q.iloc[0].vp<.5: return "stagnation",float(q.iloc[0].meta_step_fraction)
    f=d.iloc[-1]
    if f.vp>=.6:return "healthy_convergence",float(f.meta_step_fraction)
    if f.tp>=.85 and f.vp<.5:return "overfit_or_memorization",float(f.meta_step_fraction)
    return "stagnation",float(f.meta_step_fraction)

def add_dynamics(row,m,cur,init,before,g,pg,pu,window,proj,probe_stats):
    w=l2(cur); row.update(feature_weight_l2=w,feature_weight_growth_from_init=w/max(l2(init),1e-12),
        feature_weight_log_change_from_init=math.log(max(w,1e-12)/max(l2(init),1e-12)))
    if before is not None:
        u=cur-before; up=proj.project(u); window.append(up.detach().clone()); del window[:-32]
        row.update(feature_update_l2=l2(u),feature_update_to_weight_ratio=l2(u)/max(w,1e-12),feature_update_cos_prev=cosine(up,pu))
        pu=up.detach().clone()
        if len(window)>=2:
            s,_=spectral_stats(torch.stack(window),center=False,keep=0); mat=torch.stack(window)
            row.update(feature_update_effective_rank_norm=s["effective_rank_norm"],feature_update_spectral_entropy_norm=s["spectral_entropy_norm"],
                feature_update_participation_ratio_norm=s["participation_ratio_norm"],feature_update_top1_energy=s["top1_energy"],
                feature_update_directional_coherence=float(torch.linalg.vector_norm(mat.mean(0))/torch.linalg.vector_norm(mat,dim=1).mean().clamp_min(1e-12)))
    if g is not None:
        gl=l2(g); row.update(feature_grad_l2=gl,feature_grad_to_weight_ratio=gl/max(w,1e-12),feature_grad_variance=float(g.float().var(unbiased=False)),feature_grad_cos_prev=cosine(g,pg)); pg=g.detach().clone()
    row.update(probe_stats); return pg,pu

def train_generic(sp,outroot,dataroot,telemetry=True):
    seed_all(sp.seed); d=dev()
    tr,va,m,steps,lr,wd=image_setup(sp,dataroot) if sp.domain!="synthetic_sequence_gru" else seq_setup(sp)
    m=m.to(d); opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd); init=flat(m).clone(); proj=CountSketchProjector(init.numel(),128,sp.seed+999,d)
    pb=next(iter(tr))[0][:256].to(d); it=iter(tr); pg=pu=None; window=[]; rows=[]; start=time.perf_counter()
    for step in range(1,steps+1):
        try:x,y=next(it)
        except StopIteration:it=iter(tr);x,y=next(it)
        x,y=x.to(d,non_blocking=True),y.to(d,non_blocking=True); dotel=telemetry and step%20==0; before=flat(m) if dotel else None
        opt.zero_grad(set_to_none=True); m.train(); z=m(x); loss=F.cross_entropy(z,y); loss.backward(); g=grads(m) if dotel else None; opt.step()
        if step%100 and step!=1:
            if not math.isfinite(float(loss)):break
            continue
        with torch.no_grad(): zz=m(x); tl=float(F.cross_entropy(zz,y)); ta=float((zz.argmax(1)==y).float().mean())
        vl,va_acc=evaluate(m,va,d); cur=flat(m); row=dict(meta_run_id=sp.run_id,meta_domain=sp.domain,meta_step=step,meta_step_fraction=step/steps,
            feature_lr=opt.param_groups[0]["lr"],feature_train_loss=tl,feature_train_acc=ta,label_val_loss=vl,label_val_acc=va_acc)
        ps=probe(m,pb) if telemetry else {}; pg,pu=add_dynamics(row,m,cur,init,before,g,pg,pu,window,proj,ps) if telemetry else (pg,pu); rows.append(row)
        if not math.isfinite(tl):break
    df=finish_features(pd.DataFrame(rows),sp.domain); od=Path(outroot)/sp.run_id; od.mkdir(parents=True,exist_ok=True); df.to_csv(od/"metrics.csv",index=False)
    ev,et=label_event(df,sp.domain); sm=dict(run_id=sp.run_id,domain=sp.domain,intent=sp.intent,seed=sp.seed,event=ev,event_time_fraction=et,
        wall_time_sec=time.perf_counter()-start,max_steps=steps,lr=lr,weight_decay=wd,telemetry_enabled=telemetry)
    (od/"summary.json").write_text(json.dumps(sm,indent=2)); return sm

@torch.no_grad()
def eval_mod(m,x,y):
    m.eval(); n=cor=0; loss=0.
    for s in range(0,len(x),4096):
        z,_=m(x[s:s+4096],diagnostics=False); yy=y[s:s+4096]; loss+=float(F.cross_entropy(z,yy,reduction="sum")); cor+=int((z.argmax(1)==yy).sum());n+=len(yy)
    return loss/n,cor/n

def train_modular(sp,outroot,telemetry=True):
    seed_all(sp.seed); d=dev(); frac=.8 if sp.intent=="healthy" else .3; steps=20000; lr=1e-3; wd=1.
    if sp.intent=="high_lr":frac,steps,lr,wd=.5,6000,.05,0.
    if sp.intent=="low_lr":frac,steps,lr,wd=.5,8000,2e-5,.1
    if sp.intent=="small_train":frac,steps,lr,wd=.08,12000,1e-3,.2
    if sp.intent=="strong_regularization":frac,steps,lr,wd=.4,12000,5e-4,4.
    cfg=GrokConfig(seed=sp.seed,modulus=113,train_fraction=frac,d_model=128,n_heads=4,d_mlp=512,n_layers=1,lr=lr,weight_decay=wd,batch_size=0,
        max_steps=steps,eval_every=500,diagnostics_every=500,checkpoint_every=0,device=str(d))
    tx,ty,vx,vy=make_modular_dataset(113,frac,sp.seed,"add");tx,ty,vx,vy=[q.to(d) for q in (tx,ty,vx,vy)];m=ModularTransformer(cfg).to(d)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd,betas=(.9,.98));init=flat(m).clone();proj=CountSketchProjector(init.numel(),128,sp.seed+999,d)
    pg=pu=None;window=[];rows=[];start=time.perf_counter();pb=tx[:512]
    for step in range(1,steps+1):
        for q in opt.param_groups:q["lr"]=lr*min(step/max(cfg.warmup_steps,1),1.)
        dotel=telemetry and step%20==0;before=flat(m) if dotel else None;opt.zero_grad(set_to_none=True);z,_=m(tx,diagnostics=False);loss=F.cross_entropy(z,ty);loss.backward();g=grads(m) if dotel else None;opt.step()
        if step%500 and step!=1:
            if not math.isfinite(float(loss)):break
            continue
        tl,ta=eval_mod(m,tx,ty);vl,vaa=eval_mod(m,vx,vy);cur=flat(m);row=dict(meta_run_id=sp.run_id,meta_domain=sp.domain,meta_step=step,meta_step_fraction=step/steps,
            feature_lr=opt.param_groups[0]["lr"],feature_train_loss=tl,feature_train_acc=ta,label_val_loss=vl,label_val_acc=vaa)
        ps=probe_mod(m,pb) if telemetry else {};pg,pu=add_dynamics(row,m,cur,init,before,g,pg,pu,window,proj,ps) if telemetry else (pg,pu);rows.append(row)
        if not math.isfinite(tl):break
    df=finish_features(pd.DataFrame(rows),sp.domain);od=Path(outroot)/sp.run_id;od.mkdir(parents=True,exist_ok=True);df.to_csv(od/"metrics.csv",index=False)
    ev,et=label_event(df,sp.domain);sm=dict(run_id=sp.run_id,domain=sp.domain,intent=sp.intent,seed=sp.seed,event=ev,event_time_fraction=et,
        wall_time_sec=time.perf_counter()-start,max_steps=steps,lr=lr,weight_decay=wd,train_fraction=frac,telemetry_enabled=telemetry)
    (od/"summary.json").write_text(json.dumps(sm,indent=2));return sm

def run_one(sp,outroot,dataroot="/kaggle/working/tulya_data",telemetry=True):
    return train_modular(sp,outroot,telemetry) if sp.domain=="modular_transformer" else train_generic(sp,outroot,dataroot,telemetry)

def run_suite(outroot,seeds=(0,1,2,3),domains:Optional[Sequence[str]]=None,resume=True):
    ds=set(domains or DOMAINS); ss=[x for x in specs(seeds) if x.domain in ds]; root=Path(outroot);root.mkdir(parents=True,exist_ok=True); ans=[]
    for j,sp in enumerate(ss,1):
        p=root/sp.run_id/"summary.json"
        if resume and p.exists(): sm=json.loads(p.read_text()); print(f"[{j}/{len(ss)}] reuse {sp.run_id}: {sm['event']}")
        else: print(f"[{j}/{len(ss)}] run {sp.run_id}"); sm=run_one(sp,outroot); print(f" -> {sm['event']} @ {sm['event_time_fraction']:.3f}, {sm['wall_time_sec']:.1f}s")
        ans.append(sm)
    df=pd.DataFrame(ans);df.to_csv(root/"manifest.csv",index=False);return df
