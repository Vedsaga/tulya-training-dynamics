#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from experiment2_core import DOMAINS,HARMFUL

OBS=(.10,.20,.30,.40)
A=["meta_step_fraction","feature_train_loss","feature_train_acc","feature_lr"]
B=A+["feature_weight_l2","feature_weight_growth_from_init","feature_grad_l2","feature_grad_variance",
     "feature_update_l2","feature_activation_mean","feature_activation_std","feature_activation_saturation",
     "feature_prediction_entropy_norm","feature_prediction_confidence"]
C=["meta_step_fraction","feature_train_progress","feature_train_loss_log_relative","feature_weight_log_change_from_init",
   "feature_grad_to_weight_ratio","feature_update_to_weight_ratio","feature_update_cos_prev","feature_grad_cos_prev",
   "feature_update_effective_rank_norm","feature_update_spectral_entropy_norm","feature_update_participation_ratio_norm",
   "feature_update_top1_energy","feature_update_directional_coherence","feature_repr_effective_rank_norm",
   "feature_repr_spectral_entropy_norm","feature_repr_participation_ratio_norm","feature_repr_top1_energy",
   "feature_prediction_entropy_norm","feature_activation_scale_ratio"]

def prefix(df,features,obs):
    d=df[df.meta_step_fraction<=obs].sort_values("meta_step_fraction")
    if len(d)==0:d=df.iloc[[0]]
    out={"meta_observation_fraction":obs}
    for c in features:
        if c not in d:continue
        s=pd.to_numeric(d[c],errors="coerce").dropna()
        if not len(s):continue
        out[c+"__last"]=float(s.iloc[-1]);out[c+"__mean"]=float(s.mean());out[c+"__std"]=float(s.std(ddof=0))
        out[c+"__slope"]=float(np.polyfit(np.linspace(0,1,len(s)),s.to_numpy(),1)[0]) if len(s)>1 else 0.
    return out

def build_table(root):
    root=Path(root);man=pd.read_csv(root/"manifest.csv");rows=[];allf=sorted(set(A+B+C))
    for _,m in man.iterrows():
        d=pd.read_csv(root/m.run_id/"metrics.csv")
        for o in OBS:
            if float(m.event_time_fraction)<=o:continue
            r=dict(meta_run_id=m.run_id,meta_domain=m.domain,meta_observation_fraction=o,label_event=m.event,
                   label_event_time_fraction=float(m.event_time_fraction),label_time_remaining_fraction=float(m.event_time_fraction)-o,
                   label_harmful=int(m.event in HARMFUL))
            r.update(prefix(d,allf,o));rows.append(r)
    t=pd.DataFrame(rows);t.to_csv(root/"forecast_table.csv",index=False);return t

def names(base,cols):
    z=["meta_observation_fraction"]
    for c in base:
        for s in ("__last","__mean","__std","__slope"):
            if c+s in cols:z.append(c+s)
    return z

def brier(y,p,classes):
    yy=np.zeros_like(p);ix={c:i for i,c in enumerate(classes)}
    for j,v in enumerate(y):
        if v in ix:yy[j,ix[v]]=1
    return float(np.mean(np.sum((p-yy)**2,axis=1)))

def ece(y,p,classes,bins=10):
    pred=classes[p.argmax(1)];conf=p.max(1);ok=(pred==y).astype(float);ans=0.
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if m.any():ans+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return float(ans)

def evaluate(root):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler,label_binarize

    root=Path(root);t=pd.read_csv(root/"forecast_table.csv") if (root/"forecast_table.csv").exists() else build_table(root)
    systems={"A_learning_curve":A,"B_raw_telemetry":B,"C_canonical":C};rows=[]
    for held in DOMAINS:
        tr=t[t.meta_domain!=held].copy();te=t[t.meta_domain==held].copy()
        for system,base in systems.items():
            fs=names(base,t.columns);Xtr,Xte=tr[fs],te[fs];ytr,yte=tr.label_event.to_numpy(),te.label_event.to_numpy()
            clf=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                          ("m",LogisticRegression(max_iter=3000,class_weight="balanced"))])
            clf.fit(Xtr,ytr);p=clf.predict_proba(Xte);classes=clf.named_steps["m"].classes_;pred=classes[p.argmax(1)]
            shared=[c for c in classes if c in set(yte)];auc=float("nan")
            if len(shared)>=2:
                ii=[list(classes).index(c) for c in shared];pp=p[:,ii];pp/=pp.sum(1,keepdims=True).clip(min=1e-12)
                if len(shared)==2:auc=float(roc_auc_score((yte==shared[1]).astype(int),pp[:,1]))
                else:auc=float(roc_auc_score(label_binarize(yte,classes=shared),pp,average="macro",multi_class="ovr"))
            reg=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",Ridge(alpha=1.0))])
            reg.fit(Xtr,tr.label_time_remaining_fraction);tp=np.clip(reg.predict(Xte),0,1);correct=pred==yte
            tmae=float(np.mean(np.abs(tp[correct]-te.label_time_remaining_fraction.to_numpy()[correct]))) if correct.any() else float("nan")

            harmidx=[i for i,c in enumerate(classes) if c in HARMFUL]
            ptr=clf.predict_proba(Xtr)[:,harmidx].sum(1) if harmidx else np.zeros(len(tr));pte=p[:,harmidx].sum(1) if harmidx else np.zeros(len(te))
            nh=ptr[tr.label_harmful.to_numpy()==0];thr=float(np.quantile(nh,.95)) if len(nh) else 1.
            q=te[["meta_run_id","meta_observation_fraction","label_event_time_fraction","label_harmful"]].copy();q["hp"]=pte
            alerts=[]
            for rid,g in q.groupby("meta_run_id"):
                hit=g.sort_values("meta_observation_fraction");hit=hit[hit.hp>=thr]
                if len(hit):
                    x=hit.iloc[0];alerts.append((rid,int(x.label_harmful),float(x.label_event_time_fraction-x.meta_observation_fraction)))
            nonharm=int(te.groupby("meta_run_id").label_harmful.first().eq(0).sum())
            fp=sum(1 for _,h,_ in alerts if h==0);leads=[lead for _,h,lead in alerts if h==1 and lead>0]
            rows.append(dict(held_out_domain=held,system=system,macro_auroc=auc,brier=brier(yte,p,classes),ece=ece(yte,p,classes),
                event_time_mae=tmae,warning_threshold=thr,warning_fpr=fp/max(nonharm,1),
                median_warning_lead=float(np.median(leads)) if leads else float("nan"),
                n_test_prefixes=len(te),n_test_runs=te.meta_run_id.nunique(),classes_test=",".join(sorted(set(yte)))))
    r=pd.DataFrame(rows);r.to_csv(root/"evaluation_folds.csv",index=False)
    cc=r[r.system=="C_canonical"].set_index("held_out_domain");bb=r[r.system=="B_raw_telemetry"].set_index("held_out_domain")
    mean_auc=float(cc.macro_auroc.mean());min_auc=float(cc.macro_auroc.min());dauc=float((cc.macro_auroc-bb.macro_auroc).mean());worst=float((cc.macro_auroc-bb.macro_auroc).min())
    br=float((bb.brier.mean()-cc.brier.mean())/max(bb.brier.mean(),1e-12));tm=float((bb.event_time_mae.mean()-cc.event_time_mae.mean())/max(bb.event_time_mae.mean(),1e-12))
    mece=float(cc.ece.mean());mt=float(cc.event_time_mae.mean());leads=cc.median_warning_lead.dropna();ml=float(leads.median()) if len(leads) else float("nan");mf=float(cc.warning_fpr.max())
    gates=dict(mean_auc_ge_0_80=mean_auc>=.80,every_domain_auc_ge_0_75=min_auc>=.75,mean_auc_delta_ge_0_05=dauc>=.05,
        no_domain_delta_below_minus_0_02=worst>=-.02,brier_or_time_improvement=(br>=.10 or tm>=.15),mean_ece_le_0_10=mece<=.10,
        event_time_mae_le_0_15=mt<=.15,warning_fpr_le_0_05=mf<=.05,median_warning_lead_ge_0_10=math.isfinite(ml) and ml>=.10)
    core=gates["mean_auc_ge_0_80"] and gates["every_domain_auc_ge_0_75"] and gates["mean_auc_delta_ge_0_05"] and gates["no_domain_delta_below_minus_0_02"] and gates["brier_or_time_improvement"]
    s=dict(verdict="CONTINUE_TO_EXPERIMENT_3" if core else "KILL_PRODUCT_DIRECTION",
           metrics=dict(mean_auc_canonical=mean_auc,min_domain_auc_canonical=min_auc,mean_auc_delta_vs_raw=dauc,worst_domain_auc_delta_vs_raw=worst,
                        brier_relative_improvement_vs_raw=br,event_time_mae_relative_improvement_vs_raw=tm,mean_ece_canonical=mece,
                        mean_event_time_mae_canonical=mt,median_warning_lead_canonical=ml,max_warning_fpr_canonical=mf),gates=gates)
    (root/"evaluation_summary.json").write_text(json.dumps(s,indent=2));return r,s
