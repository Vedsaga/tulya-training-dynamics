#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from experiment2_core import DOMAINS,HARMFUL,NO_EVENT

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
D=sorted(set(B+C))

SYSTEMS={
    "A_learning_curve":A,
    "B_raw_telemetry":B,
    "C_canonical":C,
    "D_hybrid_diagnostic":D,
}

def prefix(df,features,obs):
    d=df[df.meta_step_fraction<=obs].sort_values("meta_step_fraction")
    if len(d)==0:d=df.iloc[[0]]
    out={"meta_observation_fraction":obs}
    for c in features:
        if c not in d:continue
        s=pd.to_numeric(d[c],errors="coerce").dropna()
        if not len(s):continue
        out[c+"__last"]=float(s.iloc[-1])
        out[c+"__mean"]=float(s.mean())
        out[c+"__std"]=float(s.std(ddof=0))
        out[c+"__slope"]=float(np.polyfit(np.linspace(0,1,len(s)),s.to_numpy(),1)[0]) if len(s)>1 else 0.
    return out

def _as_bool(x):
    if isinstance(x,(bool,np.bool_)):return bool(x)
    return str(x).strip().lower() in {"1","true","yes"}

def build_table(root):
    root=Path(root);man=pd.read_csv(root/"manifest.csv");rows=[]
    allf=sorted(set(A+B+C+D))
    for _,m in man.iterrows():
        d=pd.read_csv(root/m.run_id/"metrics.csv")
        observed=_as_bool(m.event_observed);stop=float(m.event_time_fraction)
        for o in OBS:
            if stop<=o:continue
            label=str(m.event) if observed else NO_EVENT
            r=dict(
                meta_run_id=m.run_id,meta_domain=m.domain,meta_seed=int(m.seed),meta_intent=str(m.intent),
                meta_observation_fraction=o,
                label_event=label,label_event_observed=int(observed),
                label_event_time_fraction=(stop if observed else np.nan),
                label_censor_time_fraction=(np.nan if observed else stop),
                label_time_remaining_fraction=((stop-o) if observed else np.nan),
                label_harmful=int(observed and str(m.event) in HARMFUL),
            )
            r.update(prefix(d,allf,o));rows.append(r)
    t=pd.DataFrame(rows);t.to_csv(root/"forecast_table.csv",index=False);return t

def names(base,cols):
    z=["meta_observation_fraction"]
    for c in base:
        for s in ("__last","__mean","__std","__slope"):
            if c+s in cols:z.append(c+s)
    return z

def classifier():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler()),
        ("m",LogisticRegression(max_iter=3000,class_weight="balanced")),
    ])

def regressor():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("sc",StandardScaler()),
        ("m",Ridge(alpha=1.0)),
    ])

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

def macro_auc(y,p,classes):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    present=[c for c in classes if c in set(y)]
    if len(present)<2:return float("nan")
    ii=[list(classes).index(c) for c in present]
    pp=p[:,ii];pp=pp/pp.sum(1,keepdims=True).clip(min=1e-12)
    try:
        if len(present)==2:
            return float(roc_auc_score((y==present[1]).astype(int),pp[:,1]))
        return float(roc_auc_score(label_binarize(y,classes=present),pp,average="macro",multi_class="ovr"))
    except ValueError:
        return float("nan")

def bootstrap_auc_ci(y,p,classes,run_ids,n_boot=500,seed=12345):
    rng=np.random.default_rng(seed);u=np.array(sorted(set(run_ids)));vals=[]
    if len(u)<2:return float("nan"),float("nan")
    by={rid:np.flatnonzero(run_ids==rid) for rid in u}
    for _ in range(n_boot):
        draw=rng.choice(u,size=len(u),replace=True)
        idx=np.concatenate([by[r] for r in draw])
        v=macro_auc(y[idx],p[idx],classes)
        if math.isfinite(v):vals.append(v)
    if len(vals)<20:return float("nan"),float("nan")
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def _harm_prob(p,classes):
    idx=[i for i,c in enumerate(classes) if c in HARMFUL]
    return p[:,idx].sum(1) if idx else np.zeros(len(p))

def _run_level_warning(tr,te,ptr,pte):
    """Choose threshold using max risk per training run; score first held-out alert."""
    tq=tr[["meta_run_id","meta_observation_fraction","label_harmful"]].copy()
    tq["hp"]=ptr
    train_run=tq.groupby("meta_run_id",as_index=False).agg(
        label_harmful=("label_harmful","first"),max_hp=("hp","max")
    )
    nonharm=train_run.loc[train_run.label_harmful==0,"max_hp"].to_numpy()
    if len(nonharm):
        try:q=float(np.quantile(nonharm,.95,method="higher"))
        except TypeError:q=float(np.quantile(nonharm,.95,interpolation="higher"))
        thr=float(np.nextafter(q,np.inf))
    else:
        thr=1.0

    q=te[["meta_run_id","meta_observation_fraction","label_event_time_fraction","label_harmful"]].copy()
    q["hp"]=pte
    alerts=[]
    for rid,g in q.groupby("meta_run_id"):
        g=g.sort_values("meta_observation_fraction")
        hit=g[g.hp>=thr]
        if len(hit):
            x=hit.iloc[0]
            harmful=int(x.label_harmful)
            obs=float(x.meta_observation_fraction)
            et=float(x.label_event_time_fraction) if pd.notna(x.label_event_time_fraction) else float("nan")
            alerts.append(dict(run_id=rid,harmful=harmful,obs=obs,event_time=et))

    run_truth=te.groupby("meta_run_id").label_harmful.first()
    nonharm_n=int((run_truth==0).sum())
    fp=sum(1 for x in alerts if x["harmful"]==0)
    leads=[x["event_time"]-x["obs"] for x in alerts if x["harmful"]==1 and math.isfinite(x["event_time"]) and x["event_time"]>x["obs"]]

    tp_saved=float(sum(max(x["event_time"]-x["obs"],0.) for x in alerts if x["harmful"]==1 and math.isfinite(x["event_time"])))
    fp_remaining=float(sum(max(1.-x["obs"],0.) for x in alerts if x["harmful"]==0))
    n_runs=max(int(te.meta_run_id.nunique()),1)
    econ={
        "economic_tp_saved_fraction_sum":tp_saved,
        "economic_fp_remaining_fraction_sum":fp_remaining,
        "economic_net_fraction_m1":(tp_saved-fp_remaining)/n_runs,
        "economic_net_fraction_m1_5":(tp_saved-1.5*fp_remaining)/n_runs,
        "economic_net_fraction_m2":(tp_saved-2.*fp_remaining)/n_runs,
        "economic_break_even_false_stop_multiplier":(tp_saved/fp_remaining if fp_remaining>0 else np.nan),
    }
    return thr,fp/max(nonharm_n,1),(float(np.median(leads)) if leads else float("nan")),econ

def _within_domain_diagnostics(t):
    from sklearn.model_selection import StratifiedGroupKFold
    rows=[]
    for domain in DOMAINS:
        td=t[t.meta_domain==domain].copy()
        run_labels=td.groupby("meta_run_id").label_event.first()
        if run_labels.nunique()<2:continue
        min_count=int(run_labels.value_counts().min())
        n_splits=min(4,min_count)
        if n_splits<2:continue
        for system,base in SYSTEMS.items():
            fs=names(base,t.columns)
            y=td.label_event.to_numpy();groups=td.meta_run_id.to_numpy()
            classes=np.array(sorted(td.label_event.unique()))
            oof=np.zeros((len(td),len(classes)),dtype=float)
            ok=np.zeros(len(td),dtype=bool)
            cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=20260907)
            for tridx,teidx in cv.split(td[fs],y,groups):
                clf=classifier();clf.fit(td.iloc[tridx][fs],y[tridx])
                pp=clf.predict_proba(td.iloc[teidx][fs]);cc=clf.named_steps["m"].classes_
                for j,c in enumerate(cc):
                    oof[teidx,list(classes).index(c)]=pp[:,j]
                s=oof[teidx].sum(1,keepdims=True);oof[teidx]/=np.clip(s,1e-12,None);ok[teidx]=True
            auc=macro_auc(y[ok],oof[ok],classes)
            rows.append(dict(domain=domain,system=system,in_domain_grouped_macro_auroc=auc,n_splits=n_splits,n_runs=td.meta_run_id.nunique()))
    return pd.DataFrame(rows)

def _domain_identity_diagnostics(t):
    from sklearn.model_selection import StratifiedGroupKFold
    rows=[]
    y=t.meta_domain.to_numpy();groups=t.meta_run_id.to_numpy();classes=np.array(sorted(t.meta_domain.unique()))
    for system,base in {k:v for k,v in SYSTEMS.items() if k!="A_learning_curve"}.items():
        fs=names(base,t.columns);oof=np.zeros((len(t),len(classes)));ok=np.zeros(len(t),dtype=bool)
        cv=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=20260908)
        for tridx,teidx in cv.split(t[fs],y,groups):
            clf=classifier();clf.fit(t.iloc[tridx][fs],y[tridx])
            pp=clf.predict_proba(t.iloc[teidx][fs]);cc=clf.named_steps["m"].classes_
            for j,c in enumerate(cc):
                oof[teidx,list(classes).index(c)]=pp[:,j]
            s=oof[teidx].sum(1,keepdims=True);oof[teidx]/=np.clip(s,1e-12,None);ok[teidx]=True
        rows.append(dict(system=system,domain_identity_macro_auroc=macro_auc(y[ok],oof[ok],classes),n_runs=t.meta_run_id.nunique()))
    return pd.DataFrame(rows)

def _failure_interpretation(r,within,domain_id):
    z=r.groupby("system").macro_auroc.mean().to_dict()
    A0=z.get("A_learning_curve",float("nan"))
    B0=z.get("B_raw_telemetry",float("nan"))
    C0=z.get("C_canonical",float("nan"))
    D0=z.get("D_hybrid_diagnostic",float("nan"))
    eps=.02
    if all(math.isfinite(x) for x in (A0,B0,C0,D0)):
        if C0>B0+eps and D0<=C0+eps:
            pattern="CANONICAL_APPROX_SUFFICIENT"
        elif C0>B0+eps and D0>C0+eps:
            pattern="CANONICAL_ADDS_SIGNAL_BUT_DISCARDS_SOME_RAW_INFO"
        elif B0>C0+eps and D0<=B0+eps:
            pattern="NORMALIZATION_HARMFUL_OR_REDUNDANT"
        elif D0>max(B0,C0)+eps:
            pattern="RAW_AND_NORMALIZED_COMPLEMENTARY"
        elif max(A0,B0,C0,D0)-min(A0,B0,C0,D0)<=.05:
            pattern="LITTLE_INCREMENTAL_TELEMETRY_SIGNAL"
        else:
            pattern="MIXED"
    else:
        pattern="INSUFFICIENT_METRICS"

    in_mean=within.groupby("system").in_domain_grouped_macro_auroc.mean().to_dict() if len(within) else {}
    cross_gap={}
    for s in SYSTEMS:
        if s in in_mean and s in z and math.isfinite(in_mean[s]) and math.isfinite(z[s]):
            cross_gap[s]=float(in_mean[s]-z[s])
    did=domain_id.set_index("system").domain_identity_macro_auroc.to_dict() if len(domain_id) else {}
    return dict(
        diagnostic_pattern=pattern,
        mean_zero_shot_auc={k:(None if not math.isfinite(v) else float(v)) for k,v in z.items()},
        mean_in_domain_auc={k:(None if not math.isfinite(v) else float(v)) for k,v in in_mean.items()},
        in_domain_minus_zero_shot_gap=cross_gap,
        domain_identity_auc={k:(None if not math.isfinite(v) else float(v)) for k,v in did.items()},
    )

def evaluate(root):
    root=Path(root)
    t=pd.read_csv(root/"forecast_table.csv") if (root/"forecast_table.csv").exists() else build_table(root)
    rows=[]
    for held in DOMAINS:
        tr=t[t.meta_domain!=held].copy();te=t[t.meta_domain==held].copy()
        for system,base in SYSTEMS.items():
            fs=names(base,t.columns);Xtr,Xte=tr[fs],te[fs]
            ytr,yte=tr.label_event.to_numpy(),te.label_event.to_numpy()
            clf=classifier();clf.fit(Xtr,ytr)
            p=clf.predict_proba(Xte);classes=clf.named_steps["m"].classes_;pred=classes[p.argmax(1)]
            auc=macro_auc(yte,p,classes)
            ci_lo,ci_hi=bootstrap_auc_ci(
                yte,p,classes,te.meta_run_id.to_numpy(),
                n_boot=500,seed=20260907+list(DOMAINS).index(held)
            )

            tr_time=tr[tr.label_event_observed==1]
            if len(tr_time)>=2:
                reg=regressor();reg.fit(tr_time[fs],tr_time.label_time_remaining_fraction)
                tp=np.clip(reg.predict(Xte),0,1)
                correct=(pred==yte)&(te.label_event_observed.to_numpy()==1)
                tmae=float(np.mean(np.abs(tp[correct]-te.label_time_remaining_fraction.to_numpy()[correct]))) if correct.any() else float("nan")
            else:
                tmae=float("nan")

            ptr=_harm_prob(clf.predict_proba(Xtr),classes)
            pte=_harm_prob(p,classes)
            thr,fpr,lead,econ=_run_level_warning(tr,te,ptr,pte)

            row=dict(
                held_out_domain=held,system=system,macro_auroc=auc,
                macro_auroc_ci_low=ci_lo,macro_auroc_ci_high=ci_hi,
                brier=brier(yte,p,classes),ece=ece(yte,p,classes),
                event_time_mae=tmae,warning_threshold=thr,warning_fpr=fpr,
                median_warning_lead=lead,n_test_prefixes=len(te),n_test_runs=te.meta_run_id.nunique(),
                classes_test=",".join(sorted(set(yte))),
            )
            row.update(econ);rows.append(row)

    r=pd.DataFrame(rows);r.to_csv(root/"evaluation_folds.csv",index=False)

    within=_within_domain_diagnostics(t)
    within.to_csv(root/"diagnostic_in_domain.csv",index=False)
    domain_id=_domain_identity_diagnostics(t)
    domain_id.to_csv(root/"diagnostic_domain_identity.csv",index=False)
    interpretation=_failure_interpretation(r,within,domain_id)
    (root/"diagnostic_interpretation.json").write_text(json.dumps(interpretation,indent=2))

    # Core verdict remains frozen on C vs B only. D cannot rescue it.
    cc=r[r.system=="C_canonical"].set_index("held_out_domain")
    bb=r[r.system=="B_raw_telemetry"].set_index("held_out_domain")
    mean_auc=float(cc.macro_auroc.mean());min_auc=float(cc.macro_auroc.min())
    dauc=float((cc.macro_auroc-bb.macro_auroc).mean())
    worst=float((cc.macro_auroc-bb.macro_auroc).min())
    br=float((bb.brier.mean()-cc.brier.mean())/max(bb.brier.mean(),1e-12))
    tm=float((bb.event_time_mae.mean()-cc.event_time_mae.mean())/max(bb.event_time_mae.mean(),1e-12))
    mece=float(cc.ece.mean());mt=float(cc.event_time_mae.mean())
    leads=cc.median_warning_lead.dropna()
    ml=float(leads.median()) if len(leads) else float("nan")
    mf=float(cc.warning_fpr.max())
    gates=dict(
        mean_auc_ge_0_80=mean_auc>=.80,
        every_domain_auc_ge_0_75=min_auc>=.75,
        mean_auc_delta_ge_0_05=dauc>=.05,
        no_domain_delta_below_minus_0_02=worst>=-.02,
        brier_or_time_improvement=(br>=.10 or tm>=.15),
        mean_ece_le_0_10=mece<=.10,
        event_time_mae_le_0_15=mt<=.15,
        warning_fpr_le_0_05=mf<=.05,
        median_warning_lead_ge_0_10=math.isfinite(ml) and ml>=.10,
    )
    core=(
        gates["mean_auc_ge_0_80"] and gates["every_domain_auc_ge_0_75"] and
        gates["mean_auc_delta_ge_0_05"] and gates["no_domain_delta_below_minus_0_02"] and
        gates["brier_or_time_improvement"]
    )
    s=dict(
        verdict="CONTINUE_TO_EXPERIMENT_3" if core else "KILL_PRODUCT_DIRECTION",
        metrics=dict(
            mean_auc_canonical=mean_auc,min_domain_auc_canonical=min_auc,
            mean_auc_delta_vs_raw=dauc,worst_domain_auc_delta_vs_raw=worst,
            brier_relative_improvement_vs_raw=br,
            event_time_mae_relative_improvement_vs_raw=tm,
            mean_ece_canonical=mece,mean_event_time_mae_canonical=mt,
            median_warning_lead_canonical=ml,max_warning_fpr_canonical=mf,
        ),
        gates=gates,
        diagnostics=interpretation,
    )
    (root/"evaluation_summary.json").write_text(json.dumps(s,indent=2))
    return r,s,within,domain_id
