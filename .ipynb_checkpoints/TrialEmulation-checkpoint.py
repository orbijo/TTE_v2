import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import tempfile
from scipy.stats import logistic
import matplotlib.pyplot as plt

class TrialSequence:
    def __init__(self, estimand="PP"):
        self.estimand = estimand
        self.data = None
        self.switch_weights = None
        self.censor_weights = None
        self.expansion = None
        self.outcome_model = None
        self.outcome_data_sampled = None

    def set_data(self, data, id, period, treatment, outcome, eligible):
        self.data = data.copy()
        self.data["id"] = self.data[id]
        self.data["period"] = self.data[period]
        self.data["treatment"] = self.data[treatment]
        self.data["outcome"] = self.data[outcome]
        self.data["eligible"] = self.data[eligible]
        self.data["time_on_regime"] = self.data.groupby("id")["treatment"].cumsum() - self.data["treatment"]
        return self

    def set_switch_weight_model(self, numerator, denominator, model_fitter):
        if self.estimand == "ITT":
            raise ValueError("Switch weights are not applicable for ITT estimand.")
        self.switch_weights = {
            "numerator": numerator,
            "denominator": denominator,
            "model_fitter": model_fitter,
        }
        return self

    def set_censor_weight_model(self, censor_event, numerator, denominator, pool_models="none", model_fitter=None):
        self.censor_weights = {
            "censor_event": censor_event,
            "numerator": numerator,
            "denominator": denominator,
            "pool_models": pool_models,
            "model_fitter": model_fitter or StatsGLMlogit()
        }
        return self

    def calculate_weights(self):
        if self.censor_weights:
            self._calculate_censor_weights()
        if self.switch_weights:
            self._calculate_switch_weights()
        return self

    def _calculate_censor_weights(self):
        censor_event = self.censor_weights["censor_event"]
        numerator_formula = f"1 - {censor_event} ~ " + str(self.censor_weights["numerator"])[1:]
        denominator_formula = f"1 - {censor_event} ~ " + str(self.censor_weights["denominator"])[1:]
        pool_models = self.censor_weights["pool_models"]
        model_fitter = self.censor_weights["model_fitter"]

        if pool_models == "numerator":
            numerator_model = model_fitter.fit(numerator_formula, self.data)
            self.data["wtC_num"] = numerator_model.predict(self.data)
            denominator_0 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 0])
            denominator_1 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=0) == 1])
            self.data.loc[self.data["treatment"].shift(fill_value=1) == 0, "wtC_den"] = denominator_0.predict(self.data[self.data["treatment"].shift(fill_value=1) == 0])
            self.data.loc[self.data["treatment"].shift(fill_value=0) == 1, "wtC_den"] = denominator_1.predict(self.data[self.data["treatment"].shift(fill_value=0) == 1])
        else:
            numerator_0 = model_fitter.fit(numerator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 0])
            numerator_1 = model_fitter.fit(numerator_formula, self.data[self.data["treatment"].shift(fill_value=0) == 1])
            denominator_0 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 0])
            denominator_1 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=0) == 1])
            self.data.loc[self.data["treatment"].shift(fill_value=1) == 0, "wtC_num"] = numerator_0.predict(self.data[self.data["treatment"].shift(fill_value=1) == 0])
            self.data.loc[self.data["treatment"].shift(fill_value=0) == 1, "wtC_num"] = numerator_1.predict(self.data[self.data["treatment"].shift(fill_value=0) == 1])
            self.data.loc[self.data["treatment"].shift(fill_value=1) == 0, "wtC_den"] = denominator_0.predict(self.data[self.data["treatment"].shift(fill_value=1) == 0])
            self.data.loc[self.data["treatment"].shift(fill_value=0) == 1, "wtC_den"] = denominator_1.predict(self.data[self.data["treatment"].shift(fill_value=0) == 1])

        self.data["wtC"] = self.data["wtC_num"] / self.data["wtC_den"]
        self.data["wtC"] = self.data["wtC"].fillna(1)
        return self

    def _calculate_switch_weights(self):
        numerator_formula = "treatment ~ " + str(self.switch_weights["numerator"])[1:]
        denominator_formula = "treatment ~ " + str(self.switch_weights["denominator"])[1:]
        model_fitter = self.switch_weights["model_fitter"]

        numerator_1 = model_fitter.fit(numerator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 1])
        denominator_1 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 1])
        numerator_0 = model_fitter.fit(numerator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 0])
        denominator_0 = model_fitter.fit(denominator_formula, self.data[self.data["treatment"].shift(fill_value=1) == 0])

        self.data.loc[self.data["treatment"].shift(fill_value=1) == 1, "wt_num"] = numerator_1.predict(self.data[self.data["treatment"].shift(fill_value=1) == 1])
        self.data.loc[self.data["treatment"].shift(fill_value=1) == 1, "wt_den"] = denominator_1.predict(self.data[self.data["treatment"].shift(fill_value=1) == 1])
        self.data.loc[self.data["treatment"].shift(fill_value=1) == 0, "wt_num"] = numerator_0.predict(self.data[self.data["treatment"].shift(fill_value=1) == 0])
        self.data.loc[self.data["treatment"].shift(fill_value=1) == 0, "wt_den"] = denominator_0.predict(self.data[self.data["treatment"].shift(fill_value=1) == 0])

        self.data["wt"] = self.data["wt_num"] / self.data["wt_den"]
        self.data["wt"] = self.data["wt"].fillna(1)
        return self

    def set_outcome_model(self, adjustment_terms=None):
        if adjustment_terms is None:
            self.outcome_model = {"adjustment_terms": None}
        else:
            self.outcome_model = {"adjustment_terms": adjustment_terms}
        return self

    def set_expansion_options(self, output, chunk_size):
        self.expansion = {
            "output": output,
            "chunk_size": chunk_size,
            "censor_at_switch": self.estimand == "PP",
            "first_period": 0,
            "last_period": float("inf"),
        }
        return self

    def expand_trials(self):
        chunk_size = self.expansion["chunk_size"]
        censor_at_switch = self.expansion["censor_at_switch"]
        first_period = self.expansion["first_period"]
        last_period = self.expansion["last_period"]

        expanded_data = []
        unique_ids = self.data["id"].unique()
        for i in range(0, len(unique_ids), chunk_size):
            chunk_ids = unique_ids[i:i + chunk_size]
            chunk_data = self.data[self.data["id"].isin(chunk_ids)].copy()

            for id_val in chunk_ids:
                patient_data = chunk_data[chunk_data["id"] == id_val].copy()
                max_period = patient_data["period"].max()

                for trial_period in range(first_period, min(max_period + 1, int(last_period + 1))):
                    trial_data = patient_data[patient_data["period"] >= trial_period].copy()
                    trial_data["trial_period"] = trial_period
                    trial_data["followup_time"] = trial_data["period"] - trial_period
                    trial_data["assigned_treatment"] = trial_data.loc[trial_data["period"] == trial_period, "treatment"].iloc[0]

                    if censor_at_switch:
                        switch_period = trial_data.loc[trial_data["period"] > trial_period, "treatment"].ne(trial_data.loc[trial_data["period"] == trial_period, "treatment"].iloc[0]).idxmax()
                        if pd.notna(switch_period):
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["id"]==id_val, "period"].max()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["period"]==patient_data.loc[patient_data["id"]==id_val, "period"].max(), "period"].min()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["id"]==id_val, "period"].max()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["period"]==patient_data.loc[patient_data["id"]==id_val, "period"].max(), "period"].min()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["id"]==id_val, "period"].max()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["period"]==patient_data.loc[patient_data["id"]==id_val, "period"].max(), "period"].min()]
                            trial_data = trial_data[trial_data["period"] <= patient_data.loc[patient_data["id"]==id_val, "period"].max()]
                            trial_data = trial_data[trial_data["period"]==patient_data.loc[patient_data["id"]==id_val, "period"].max()].min()
                            switch_period_val = trial_data.loc[trial_data["period"]== patient_data.loc[patient_data["id"]==id_val, "period"].max()].index.min()
                            trial_data = trial_data.loc[:switch_period_val-1]

                    expanded_data.append(trial_data)

        self.expansion["data"] = pd.concat(expanded_data, ignore_index=True)
        return self

    def load_expanded_data(self, seed=None, p_control=1.0):
        if seed is not None:
            np.random.seed(seed)

        expanded_data = self.expansion["data"].copy()
        control_data = expanded_data[expanded_data["outcome"] == 0].copy()

        if p_control < 1.0:
            control_indices = np.random.choice(control_data.index, size=int(len(control_data) * p_control), replace=False)
            control_data = control_data.loc[control_indices]

        case_data = expanded_data[expanded_data["outcome"] == 1].copy()
        sampled_data = pd.concat([case_data, control_data], ignore_index=True)

        sampled_data["sample_weight"] = 1
        sampled_data.loc[sampled_data["outcome"] == 0, "sample_weight"] = 1 / p_control

        if self.censor_weights:
            sampled_data["w"] = sampled_data["wtC"] * sampled_data["sample_weight"]
        if self.switch_weights:
            sampled_data["w"] = sampled_data["wt"] * sampled_data["sample_weight"]
        if self.censor_weights and self.switch_weights:
            sampled_data["w"] = sampled_data["wt"] * sampled_data["wtC"] * sampled_data["sample_weight"]

        self.outcome_data_sampled = sampled_data
        return self

    def fit_msm(self, weight_cols, modify_weights=None):
        data = self.outcome_data_sampled.copy()
        formula = "outcome ~ assigned_treatment"
        if self.outcome_model and self.outcome_model["adjustment_terms"] is not None:
            formula += " + " + str(self.outcome_model["adjustment_terms"])[1:]
        formula += " + followup_time + I(followup_time**2) + trial_period + I(trial_period**2)"

        weights = data[weight_cols].product(axis=1)

        if modify_weights:
            weights = modify_weights(weights)

        model = StatsGLMlogit().fit(formula, data, weights=weights)

        self.outcome_model["fitted"] = model
        self.outcome_model["formula"] = formula
        self.outcome_model["treatment_variable"] = "assigned_treatment"
        if self.outcome_model and self.outcome_model["adjustment_terms"] is not None:
            self.outcome_model["adjustment_variables"] = str(self.outcome_model["adjustment_terms"])[1:]
        self.outcome_model["model_fitter"] = "te_stats_glm_logit"
        return self

    def predict(self, newdata, predict_times, type="survival"):
        results = {}
        treatment_values = newdata["assigned_treatment"].unique()
        followup_times = predict_times
        survival_results = {}
        for treatment_val in treatment_values:
            treatment_data = newdata.copy()
            treatment_data["assigned_treatment"] = treatment_val
            survival_probabilities = []
            for time in followup_times:
                predict_data = treatment_data.copy()
                predict_data["followup_time"] = time
                predictions = self.outcome_model["fitted"].predict(predict_data)
                if type == "survival":
                    survival_prob = 1 - predictions
                else:
                    survival_prob = predictions
                survival_probabilities.append(survival_prob.mean())
            survival_results[treatment_val] = survival_probabilities

        survival_df = pd.DataFrame(survival_results)
        survival_df["followup_time"] = followup_times
        results["survival"] = survival_df

        if len(treatment_values) == 2:
            diff = survival_df[treatment_values[1]] - survival_df[treatment_values[0]]
            se = np.sqrt(survival_df[treatment_values[1]].var() + survival_df[treatment_values[0]].var())
            ci_low = diff - 1.96 * se
            ci_high = diff + 1.96 * se
            diff_df = pd.DataFrame({
                "followup_time": followup_times,
                "survival_diff": diff,
                "2.5%": ci_low,
                "97.5%": ci_high
            })
            results["difference"] = diff_df
        return results
        
def outcome_data(trial_obj):
    return trial_obj.outcome_data_sampled.copy()

class StatsGLMlogit:
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.model = None

    def fit(self, formula, data, weights=None):
        model_ols = ols(formula, data)
        model_formula = model_ols.formula
        y, X = sm.formula.dmatrices(model_formula, data, return_type='dataframe') # Correct access
        if weights is not None:
            self.model = sm.GLM(y, X, family=sm.families.Binomial(), weights=weights).fit()
        else:
            self.model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return self

    def predict(self, data):
        model_ols = ols(self.model.formula, data)
        model_formula = model_ols.formula
        y, X = sm.formula.dmatrices(model_formula, data, return_type='dataframe') # Correct access
        return self.model.predict(X)

    def summary(self):
        return self.model.summary()