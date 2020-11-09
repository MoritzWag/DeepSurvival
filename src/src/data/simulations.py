import numpy as np 
import pandas as pd
import pdb


def make_risk_score_for_groups(y, 
                               n_groups, 
                               seed):
    """
    """
    random = np.random.RandomState(seed)

    classes = np.unique(y)
    group_assignment = {}
    group_members = {}
    groups = random.randint(n_groups, size=classes.shape)
    for label, group in zip(classes, groups):
        group_assignment[label] = group
        group_members.setdefault(group, []).append(label)
    
    risk_per_class = {}
    for label in classes: 
        group_idx = group_assignment[label]
        group = group_members[group_idx]
        labeled_idx = group.index(label)
        group_size = len(group)

        risk_score = np.sqrt(group_idx + 1e-4) * 1.75
        risk_score -= (labeled_idx - (group_size // 2)) / 25. 
        risk_per_class[label] = risk_score

    assignment = pd.concat((
                            pd.Series(risk_per_class, name='risk_score'),
                            pd.Series(group_assignment, name='risk_group')
    ), axis=1).rename_axis('class_label')

    risk_score = np.array([risk_per_class[yy] for yy in y])

    return assignment, risk_score


def generate_survival_time(num_samples,
                           mean_survival_time,
                           prob_censored,
                           risk_score,
                           seed):
    """
    """
    random = np.random.RandomState(seed)
    baseline_hazard = 1. / mean_survival_time
    scale = baseline_hazard * np.exp(risk_score)
    u = random.uniform(low=0, high=1, size=risk_score.shape[0])
    t = -np.log(u) / scale

    # generate time of censoring
    qt = np.quantile(t, 1.0 - prob_censored)
    c =  random.uniform(low=t.min(), high=qt)

    # apply censoring 
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event