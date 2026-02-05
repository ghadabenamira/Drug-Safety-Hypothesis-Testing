
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import pingouin
import seaborn as sns
import matplotlib.pyplot as plt

drug_safety = pd.read_csv("drug_safety.csv")

adv_eff_ct = pd.crosstab(drug_safety["trx"], drug_safety["adverse_effects"])

print("Crosstab index:", adv_eff_ct.index)

yess = [adv_eff_ct.loc["Drug", "Yes"], adv_eff_ct.loc["Placebo", "Yes"]]

n = [adv_eff_ct.loc["Drug"].sum(), adv_eff_ct.loc["Placebo"].sum()]

two_sample_results = proportions_ztest(yess, n)
two_sample_p_value = two_sample_results[1]

num_effects_groups = pingouin.chi2_independence(data=drug_safety, x="num_effects", y="trx")

num_effects_p_value = num_effects_groups[2]["pval"][0]

sns.histplot(data=drug_safety, x="age", hue="trx")

normality= pingouin.normality( data = drug_safety, dv='age', group='trx', method='shapiro', alpha= 0.05 )

age_trx = drug_safety.loc[drug_safety["trx"] == "Drug", "age"]

age_placebo = drug_safety.loc[drug_safety["trx"] == "Placebo", "age"]

age_group_effects = pingouin.mwu(age_trx, age_placebo)

age_group_effects_p_value = age_group_effects["p-val"]
