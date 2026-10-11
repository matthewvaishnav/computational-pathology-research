from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.scorpion import run_pa_nf_v5_rt1_scorpion_translation as rt1
from experiments.scorpion import run_pa_nf_v5_rt3_residual_tangent_regularization as rt3
from experiments.scorpion import run_pa_nf_v5_rt4_identity_margin_factorial as rt4


class IdentityEncoderStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Identity()
    def invert_operator(self, x, scanner_index):
        return x


def _toy_bundle(n=12,seed=24):
    generator=np.random.default_rng(seed)
    obs=generator.normal(size=(n,len(rt1.SCANNERS),32)).astype(np.float32)
    slides=np.asarray([f"slide_{i//3:02d}" for i in range(n)])
    regions=np.asarray([f"region_{i%3:02d}" for i in range(n)])
    return rt1.SplitBundle(obs,slides,regions)


def _known(heldout="GT450"):
    index=rt1.SCANNER_TO_INDEX[heldout]
    return tuple(i for i in range(len(rt1.SCANNERS)) if i!=index)


def test_five_capacity_matched_arms_and_new_seeds():
    config=rt1.RT1Config()
    states=[]
    for arm in rt4.ARMS:
        v4.set_deterministic_seed(rt4.MODEL_SEEDS[0])
        model=rt1.RT1ReferenceGaugeModel(config,rt3._arm_family(rt4._base_arm(arm)))
        assert rt1.parameter_count(model)==9640
        states.append(model.state_dict())
    for other in states[1:]:
        assert set(other)==set(states[0])
        for key in other:
            torch.testing.assert_close(other[key],states[0][key],rtol=0,atol=0)
    assert set(rt4.MODEL_SEEDS).isdisjoint(set(rt3.MODEL_SEEDS))
    assert len(rt4.ARMS)*len(rt4.MODEL_SEEDS)*len(rt1.FOLDS)*len(rt1.HELDOUT_SCANNERS)==300


def test_train_region_unique_identity_required():
    bundle=_toy_bundle()
    rt4._check_training_region_identity(bundle)
    corrupted=rt1.SplitBundle(
        observations=bundle.observations,
        slide_ids=bundle.slide_ids.copy(),
        region_ids=bundle.region_ids.copy()
    )
    corrupted.region_ids[1]=corrupted.region_ids[0]
    corrupted.slide_ids[1]=corrupted.slide_ids[0]
    with pytest.raises(rt4.RT4Error,match="unique"):
        rt4._check_training_region_identity(corrupted)


def test_matched_identity_better_than_mispairing():
    n=8
    data=torch.zeros((n,len(rt1.SCANNERS),n),dtype=torch.float32)
    eye=torch.eye(n)
    for s in range(len(rt1.SCANNERS)):
        data[:,s,:]=eye
    model=IdentityEncoderStub()
    positive=rt4.paired_identity_loss(model,data,(0,1,2))
    wrong=data.clone()
    wrong[:,1,:]=data[torch.arange(n).roll(1),1,:]
    wrong[:,2,:]=data[torch.arange(n).roll(2),2,:]
    negative=rt4.paired_identity_loss(model,wrong,(0,1,2))
    assert positive.item()<0.001
    assert negative.item()>positive.item()+3


def test_identity_gradient_only_flows_to_encoder_and_not_scanner_operators():
    bundle=_toy_bundle()
    v4.set_deterministic_seed(5001)
    model=rt1.RT1ReferenceGaugeModel(rt1.RT1Config(),"inverse_transport")
    obs=torch.tensor(bundle.observations)
    known=_known()
    penalty=rt4.paired_identity_loss(model,obs,known)
    assert torch.isfinite(penalty)
    penalty.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad).item()>0
               for p in model.encoder.parameters())
    for s in known:
        if s!=rt1.REFERENCE_INDEX:
            assert all(p.grad is None or torch.count_nonzero(p.grad)==0
                       for p in model.operator_module(s).parameters())


def test_original_arms_exactly_delegate_to_frozen_rt3_trainer():
    bundle=_toy_bundle()
    known=_known()
    config=replace(rt1.RT1Config(),epochs=2)
    device=torch.device("cpu")
    units=rt3._unit_random_directions(bundle,known,fold=0,heldout_index=1,seed=5001,device=device)
    v4.set_deterministic_seed(5001)
    original=rt1.RT1ReferenceGaugeModel(config,"inverse_transport")
    v4.set_deterministic_seed(5001)
    rt4_model=rt1.RT1ReferenceGaugeModel(config,"inverse_transport")
    expected=rt3.train_arm(original,rt3.ARM_INVERSE_ISOTROPIC,bundle,known,config,device,units)
    observed=rt4.train_arm(rt4_model,rt4.ARM_ISO,bundle,known,config,device,units)
    for a,b in zip(expected["history"],observed["history"]):
        for key,val in a.items():
            assert b[key]==val
        assert b["paired_identity_loss"]==0
    for key,val in original.state_dict().items():
        torch.testing.assert_close(val,rt4_model.state_dict()[key],rtol=0,atol=0)


def test_identity_enabled_train_two_epochs_updates_encoder():
    bundle=_toy_bundle()
    known=_known()
    config=replace(rt1.RT1Config(),epochs=2)
    device=torch.device("cpu")
    units=rt3._unit_random_directions(bundle,known,fold=0,heldout_index=1,seed=5001,device=device)
    v4.set_deterministic_seed(5001)
    model=rt1.RT1ReferenceGaugeModel(config,"inverse_transport")
    before={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
    out=rt4.train_arm(model,rt4.ARM_RES_ID,bundle,known,config,device,units)
    assert out["history"][-1]["paired_identity_loss"]>0
    assert all(np.isfinite(list(row.values())).all() for row in out["history"])
    assert any(not torch.equal(before[k],v) for k,v in model.encoder.state_dict().items())


def _synthetic_fit_rows():
    # A slide only occurs in one fold, as in SCORPION. Three model seeds and
    # four scanner holdouts revisit that SAME biological slide.
    rows=[]
    counts=(10,10,10,9,9)
    offset=0
    for fold,count in enumerate(counts):
        test_slides=[f"s{j:02d}" for j in range(offset,offset+count)]
        offset+=count
        for heldout in rt1.HELDOUT_SCANNERS:
            for seed in rt4.MODEL_SEEDS:
                for arm in rt4.ARMS:
                    if arm==rt4.ARM_CONTROL:
                        align,ret,probe,margin=0.10,0.85,0.30,0.06
                    elif arm==rt4.ARM_ISO:
                        align,ret,probe,margin=0.030,0.70,0.30,0.04
                    elif arm==rt4.ARM_ISO_ID:
                        align,ret,probe,margin=0.032,0.86,0.21,0.10
                    elif arm==rt4.ARM_RES:
                        align,ret,probe,margin=0.025,0.80,0.26,0.08
                    else:
                        align,ret,probe,margin=0.028,0.89,0.24,0.11
                    metrics=[]
                    gallery={}
                    for slide in test_slides:
                        metrics.append({
                            "slide_id":slide,
                            "heldout_alignment_mse":align,
                            "heldout_retrieval_top1":ret,
                            "known_scanner_probe_accuracy":probe,
                            "reference_reconstruction_mse":0.7,
                            "operator_only_heldout_transport_gain":0.03,
                            "operator_only_known_transport_gain":0.05,
                        })
                        gallery[slide]={
                            "fold_global_top1":ret,
                            "cross_slide_cosine_margin":margin,
                            "within_slide_cosine_margin":margin,
                            "matched_cross_slide_nine_strict_top1":ret,
                            "cross_slide_only_error_rate":0.1,
                        }
                    rows.append({
                        "arm":arm,"heldout_scanner":heldout,"seed":seed,"fold":fold,
                        "parameter_count":9640,
                        "heldout_operator_max_abs_diff_vs_control":0.,
                        "evaluation":{
                            "slide_metrics":metrics,
                            "max_operator_inverse_roundtrip_mse":1e-12,
                        },
                        "gallery_slide_metrics":gallery,
                    })
    assert len(rows)==300
    return rows


def test_slide_level_aggregate_primary_gate_and_interaction():
    config=replace(rt1.RT1Config(),bootstrap_draws=100)
    outcome=rt4._aggregate(_synthetic_fit_rows(),config)
    assert outcome["independent_slide_count"]==48
    checks=outcome["frozen_development_checks"]
    assert checks["all_frozen_development_checks_pass"] is True
    ci=outcome["slide_level_intervals"]
    assert ci["iso_id_minus_iso_global_retrieval"]["mean"]==pytest.approx(0.16)
    assert ci["identity_gain_difference_in_differences_global"]["mean"]==pytest.approx(0.07)
    assert ci["iso_id_minus_iso_alignment_mse"]["mean"]==pytest.approx(0.002)


def test_incomplete_arm_aggregation_fails_closed():
    rows=_synthetic_fit_rows()
    rows.pop()
    with pytest.raises(rt4.RT4Error,match="Expected 300"):
        rt4._aggregate(rows,replace(rt1.RT1Config(),bootstrap_draws=100))
