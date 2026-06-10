#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS=(0.45,0.50,0.55,0.60)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',type=Path,default=Path('results/pathoalign_two_resource_phase_map'))
    p.add_argument('--out-dir',type=Path,default=Path('results/pathoalign_two_resource_analysis'))
    p.add_argument('--thresholds',type=float,nargs='+',default=list(DEFAULT_THRESHOLDS))
    return p.parse_args()

def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def boundary_metrics(frame:pd.DataFrame,threshold:float)->dict:
    ordered=frame.sort_values('pair_count').reset_index(drop=True).copy()
    scores=ordered['universal_biological_score_mean'].astype(float)
    passed=scores>=threshold
    passing=np.flatnonzero(passed.to_numpy())
    first=np.nan; sustained=np.nan; post=np.nan
    first_lower=np.nan; sustained_lower=np.nan
    if len(passing):
        i=int(passing[0]); first=int(ordered.loc[i,'pair_count']); first_lower=0 if i==0 else int(ordered.loc[i-1,'pair_count']); post=int((~passed.iloc[i:]).sum())
    for i in range(len(ordered)):
        if bool(passed.iloc[i]) and bool(passed.iloc[i:].all()):
            sustained=int(ordered.loc[i,'pair_count']); sustained_lower=0 if i==0 else int(ordered.loc[i-1,'pair_count']); break
    x=np.log1p(ordered['pair_count'].astype(float).to_numpy()); y=scores.to_numpy()
    auc=float(np.trapezoid(y,x)/(x.max()-x.min())) if len(x)>1 and x.max()>x.min() else float(y[0])
    best_i=int(scores.idxmax())
    max_pair=int(ordered['pair_count'].max())
    def interval(lower,val):
        if pd.isna(val): return f'> {max_pair}'
        if val==0: return '[0, 0]'
        return f'({int(lower)}, {int(val)}]'
    return {
        'threshold':threshold,
        'first_crossing':first,
        'first_boundary_interval':interval(first_lower,first),
        'sustained_crossing':sustained,
        'sustained_boundary_interval':interval(sustained_lower,sustained),
        'post_crossing_failures':post,
        'recovery_curve_auc_log_pairs':auc,
        'best_pair_count':int(ordered.loc[best_i,'pair_count']),
        'best_score':float(scores.loc[best_i]),
        'final_score':float(scores.iloc[-1]),
    }

def compute_boundaries(summary,thresholds):
    rows=[]; cols=['n','overlap','nonlinear','method']
    for key,frame in summary.groupby(cols,sort=True):
        for t in thresholds:
            row=dict(zip(cols,key)); row.update(boundary_metrics(frame,t)); rows.append(row)
    return pd.DataFrame(rows)


METHOD_LEVELS = (
    "factorized",
    "pair_consistency",
    "operator",
    "hybrid_curriculum",
)


def design_matrix(df, name):
    """Construct a deterministic design matrix for every CV split.

    Method dummy columns use a fixed global ordering so training and
    held-out matrices always have identical dimensions.
    """
    columns = [
        np.ones(len(df), dtype=float),
    ]
    column_names = [
        "intercept",
    ]

    if name in {"power", "power_overlap", "full"}:
        columns.append(
            np.log(df["n"].astype(float).to_numpy())
        )
        column_names.append("log_n")

    if name in {"power_overlap", "full"}:
        columns.append(
            df["overlap"].astype(float).to_numpy()
        )
        column_names.append("overlap")

        nonlinear = (
            df["nonlinear"]
            .astype(str)
            .str.lower()
            .map({
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
            })
        )

        if nonlinear.isna().any():
            bad_values = sorted(
                df.loc[nonlinear.isna(), "nonlinear"]
                .astype(str)
                .unique()
                .tolist()
            )
            raise ValueError(
                "Unrecognized nonlinear values: "
                f"{bad_values}"
            )

        columns.append(nonlinear.to_numpy(dtype=float))
        column_names.append("nonlinear")

    if name == "full":
        methods = df["method"].astype(str)

        # factorized is the reference category.
        for method in METHOD_LEVELS[1:]:
            columns.append(
                (methods == method).to_numpy(dtype=float)
            )
            column_names.append(f"method_{method}")

    return np.column_stack(columns), column_names


def fit_model(df,name):
    y=np.log(df['sustained_crossing'].astype(float).to_numpy()); X,names=design_matrix(df,name)
    beta,*_=np.linalg.lstsq(X,y,rcond=None); pred=X@beta
    sq=[]
    for _,test in df.groupby(['overlap','nonlinear','method']):
        train=df.drop(index=test.index)
        if len(train)<=X.shape[1]: continue
        Xt,_=design_matrix(train,name); yt=np.log(train['sustained_crossing'].astype(float).to_numpy())
        b,*_=np.linalg.lstsq(Xt,yt,rcond=None); Xv,_=design_matrix(test,name); yv=np.log(test['sustained_crossing'].astype(float).to_numpy()); sq += ((Xv@b-yv)**2).tolist()
    return {'model':name,'n_rows':len(df),'train_log_rmse':float(np.sqrt(np.mean((pred-y)**2))),'leave_condition_out_log_rmse':float(np.sqrt(np.mean(sq))) if sq else np.nan,'coefficients':json.dumps({k:float(v) for k,v in zip(names,beta)},sort_keys=True)}

def main():
    a=parse_args(); a.out_dir.mkdir(parents=True,exist_ok=True)
    summary_path=a.input_dir/'summary.csv'; phase_path=a.input_dir/'two_resource_phase_map.csv'; config_path=a.input_dir/'config.json'
    if not summary_path.exists(): raise FileNotFoundError(summary_path)
    summary=pd.read_csv(summary_path)
    boundaries=compute_boundaries(summary,a.thresholds); boundaries.to_csv(a.out_dir/'robust_recovery_boundaries.csv',index=False)
    sensitivity=boundaries.pivot_table(index=['n','overlap','nonlinear','method'],columns='threshold',values='sustained_crossing',aggfunc='first').reset_index(); sensitivity.to_csv(a.out_dir/'threshold_sensitivity.csv',index=False)
    usable=boundaries[(boundaries.threshold==0.5)&boundaries.sustained_crossing.notna()&(boundaries.method!='factorized')].copy()
    scaling=pd.DataFrame([fit_model(usable,n) for n in ['constant','power','power_overlap','full']]).sort_values('leave_condition_out_log_rmse'); scaling.to_csv(a.out_dir/'scaling_model_comparison.csv',index=False)
    manifest={'analysis_script_sha256':sha256_file(Path(__file__)),'inputs':{},'thresholds':a.thresholds,'primary_endpoint':'universal_biological_score_mean','primary_threshold':0.5,'boundary_rule':'sustained crossing'}
    for p in [summary_path,phase_path,config_path]:
        if p.exists(): manifest['inputs'][str(p)]={'sha256':sha256_file(p),'size_bytes':p.stat().st_size}
    (a.out_dir/'frozen_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (a.out_dir/'report.md').write_text('# PathoAlign v5 freeze-and-analyze report\n\nPrimary rule: sustained recovery at threshold 0.50.\n\nNearby thresholds 0.45, 0.55, and 0.60 are sensitivity analyses.\n',encoding='utf-8')
    print(f'Wrote analysis artifacts to {a.out_dir}')
if __name__=='__main__': main()
