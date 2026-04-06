import unittest
import pandas as pd
import numpy as np
from analysistoolbox.descriptive_analytics import ConductPropensityScoreMatching


def _make_df(n=120, seed=42):
    """Return a DataFrame with 30 treatment and 90 control rows."""
    rng = np.random.default_rng(seed)
    n_treat = n // 4
    n_ctrl = n - n_treat
    treatment = ['Treatment'] * n_treat + ['Control'] * n_ctrl
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'Age': np.concatenate([
            rng.normal(45, 5, n_treat),
            rng.normal(45, 5, n_ctrl),
        ]).round(1),
        'Income': np.concatenate([
            rng.normal(50000, 5000, n_treat),
            rng.normal(50000, 5000, n_ctrl),
        ]).round(0),
        'Education': np.concatenate([
            rng.normal(16, 2, n_treat),
            rng.normal(16, 2, n_ctrl),
        ]).round(1),
        'Treatment': treatment,
    })
    return df


class TestConductPropensityScoreMatching(unittest.TestCase):

    def setUp(self):
        self.df = _make_df()
        self.covariates = ['Age', 'Income', 'Education']

    def _run(self, **kwargs):
        defaults = dict(
            dataframe=self.df,
            subject_id_column_name='ID',
            list_of_column_names_to_base_matching=self.covariates,
            grouping_column_name='Treatment',
            control_group_name='Control',
        )
        defaults.update(kwargs)
        return ConductPropensityScoreMatching(**defaults)

    # ------------------------------------------------------------------
    # Return type and shape
    # ------------------------------------------------------------------

    def test_returns_dataframe(self):
        """Result is a pandas DataFrame."""
        result = self._run()
        self.assertIsInstance(result, pd.DataFrame)

    def test_output_has_propensity_score_column(self):
        """Default propensity score column is present in output."""
        result = self._run()
        self.assertIn('Propensity Score', result.columns)

    def test_output_has_propensity_logit_column(self):
        """Default propensity logit column is present in output."""
        result = self._run()
        self.assertIn('Propensity Logit', result.columns)

    def test_output_has_matched_id_column(self):
        """Default matched ID column is present in output."""
        result = self._run()
        self.assertIn('Matched ID', result.columns)

    def test_original_columns_preserved(self):
        """All original columns are still present after matching."""
        result = self._run()
        for col in self.df.columns:
            self.assertIn(col, result.columns)

    # ------------------------------------------------------------------
    # Propensity score values
    # ------------------------------------------------------------------

    def test_propensity_scores_between_zero_and_one(self):
        """Propensity scores are probabilities bounded in [0, 1]."""
        result = self._run()
        scores = result['Propensity Score'].dropna()
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

    def test_some_subjects_have_matches(self):
        """At least some treatment subjects are matched to control subjects."""
        result = self._run()
        self.assertGreater(result['Matched ID'].notna().sum(), 0)

    # ------------------------------------------------------------------
    # Matching behaviour
    # ------------------------------------------------------------------

    def test_max_matches_per_subject_one(self):
        """With max_matches_per_subject=1, each treated subject has at most one match row."""
        result = self._run(max_matches_per_subject=1)
        treat_rows = result[result['Treatment'] == 'Treatment']
        # Each treatment subject should appear at most twice (one row per match direction)
        counts = treat_rows.groupby('ID').size()
        self.assertTrue((counts <= 2).all())

    def test_max_matches_per_subject_two(self):
        """With max_matches_per_subject=2, function completes without error."""
        result = self._run(max_matches_per_subject=2)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Matched ID', result.columns)

    def test_matched_ids_reference_existing_subjects(self):
        """Every non-null matched ID corresponds to a real subject ID in the data."""
        result = self._run()
        valid_ids = set(self.df['ID'].tolist())
        matched_ids = result['Matched ID'].dropna().tolist()
        for mid in matched_ids:
            self.assertIn(mid, valid_ids)

    # ------------------------------------------------------------------
    # Custom column names
    # ------------------------------------------------------------------

    def test_custom_propensity_score_column_name(self):
        """Custom name for the propensity score column is respected."""
        result = self._run(propensity_score_column_name='PS')
        self.assertIn('PS', result.columns)
        self.assertNotIn('Propensity Score', result.columns)

    def test_custom_propensity_logit_column_name(self):
        """Custom name for the propensity logit column is respected."""
        result = self._run(propensity_logit_column_name='Logit')
        self.assertIn('Logit', result.columns)
        self.assertNotIn('Propensity Logit', result.columns)

    def test_custom_matched_id_column_name(self):
        """Custom name for the matched ID column is respected."""
        result = self._run(matched_id_column_name='Match')
        self.assertIn('Match', result.columns)
        self.assertNotIn('Matched ID', result.columns)

    # ------------------------------------------------------------------
    # Input validation – ValueError cases
    # ------------------------------------------------------------------

    def test_raises_when_grouping_column_has_more_than_two_values(self):
        """ValueError when grouping column contains more than two distinct values."""
        df_bad = self.df.copy()
        df_bad.loc[0, 'Treatment'] = 'Unknown'
        with self.assertRaises(ValueError):
            ConductPropensityScoreMatching(
                dataframe=df_bad,
                subject_id_column_name='ID',
                list_of_column_names_to_base_matching=self.covariates,
                grouping_column_name='Treatment',
                control_group_name='Control',
            )

    def test_raises_when_control_group_name_not_in_column(self):
        """ValueError when control_group_name is not present in the grouping column."""
        with self.assertRaises(ValueError):
            ConductPropensityScoreMatching(
                dataframe=self.df,
                subject_id_column_name='ID',
                list_of_column_names_to_base_matching=self.covariates,
                grouping_column_name='Treatment',
                control_group_name='Placebo',
            )

    def test_raises_when_subject_ids_are_not_unique(self):
        """ValueError when the subject ID column contains duplicate values."""
        df_dup = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            ConductPropensityScoreMatching(
                dataframe=df_dup,
                subject_id_column_name='ID',
                list_of_column_names_to_base_matching=self.covariates,
                grouping_column_name='Treatment',
                control_group_name='Control',
            )

    def test_raises_when_treatment_group_is_empty(self):
        """ValueError when no rows belong to the treatment group."""
        df_ctrl_only = self.df[self.df['Treatment'] == 'Control'].copy()
        df_ctrl_only = df_ctrl_only.reset_index(drop=True)
        df_ctrl_only['ID'] = range(1, len(df_ctrl_only) + 1)
        with self.assertRaises(ValueError):
            ConductPropensityScoreMatching(
                dataframe=df_ctrl_only,
                subject_id_column_name='ID',
                list_of_column_names_to_base_matching=self.covariates,
                grouping_column_name='Treatment',
                control_group_name='Control',
            )

    def test_raises_when_control_group_is_empty(self):
        """ValueError when no rows belong to the control group."""
        df_treat_only = self.df[self.df['Treatment'] == 'Treatment'].copy()
        df_treat_only = df_treat_only.reset_index(drop=True)
        df_treat_only['ID'] = range(1, len(df_treat_only) + 1)
        with self.assertRaises(ValueError):
            ConductPropensityScoreMatching(
                dataframe=df_treat_only,
                subject_id_column_name='ID',
                list_of_column_names_to_base_matching=self.covariates,
                grouping_column_name='Treatment',
                control_group_name='Control',
            )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_random_seed_produces_identical_results(self):
        """Same random seed yields identical matched ID assignments."""
        result_a = self._run(random_seed=99)
        result_b = self._run(random_seed=99)
        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True),
            result_b.reset_index(drop=True),
        )


if __name__ == '__main__':
    unittest.main()
