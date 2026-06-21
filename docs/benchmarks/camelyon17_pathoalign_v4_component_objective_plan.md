# PathoAlign v4 hypothesis

v3 fixed the nuisance center-loss wiring, but stronger adversary smoke remained negative.

Conclusion:
- z_nuisance can learn center/source identity.
- decoded nuisance component does not remove center/source identity from x.
- nuisance representation remains tumor-predictive.

v4 hypothesis:
Train the decoded nuisance component directly, not only z_nuisance.

Required objective changes:
- add center head on nuisance_component
- add tumor adversary on nuisance_component
- keep cleaned-feature tumor preservation
- evaluate cleaned_features, z_nuisance, and nuisance_component separately

Do not scale v3 further until component-level separation works.
