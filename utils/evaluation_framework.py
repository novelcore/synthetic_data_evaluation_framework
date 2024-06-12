import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wasserstein_distance
from sklearn.model_selection import cross_validate, StratifiedKFold
from utils.non_parametric_tests import statistical_analysis, friedman_aligned_ranks_test
from sklearn.ensemble import IsolationForest
from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import NewRowSynthesis


class EvaluationFramework:
    """
    Class for evaluating the similarity between real and synthetic datasets using various metrics.
    """

    def __init__(
        self,
        df_real_data: pd.DataFrame = None,
        d_synthetic_data: dict = None,
        categorical_features: list = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the EvaluationProtocol instance.

        Parameters
        -----------
        df_real_data (pd.DataFrame):
            The real dataset.
        d_synthetic_data (dict):
            keys (String): name of dynthesizer
            values (pd.DataFrame): the synthetic dataset.
        verbose(bool):
            Flag for presenting results
        """
        self.categorical_features = categorical_features
        self.df_real_data = df_real_data
        self.d_synthetic_data = d_synthetic_data
        self.verbose = verbose
        self.evaluation_metric_scores = dict()

    def get_synthesizers_ranking(self):
        """
        Apply the Finner test to compare the total number of score for each synthetic dataset
        Finner test is a Post-hoc non-parametric test in order to compare pairs of synthetic data, which are not biased by the initial distribution of the features

        Returns
        -------
            The Ranking DataFrame contains the methods, their FAR (Friedman Aligned Ranks), and is sorted based on FAR.
            The Finner DataFrame contains comparisons, adjusted p-values, and whether the null hypothesis is rejected.
        """
        df_scores = pd.DataFrame(self.evaluation_metric_scores).transpose()
        return statistical_analysis(df_scores, self.verbose)

    def wasserstein_cramers_v_test(self) -> dict[str, float]:
        """
        Apply the Wasserstein distance to compare the distributions of real and synthetic datasets to the continuous features
        and Cramer's V to compare the distributions of real and synthetic datasets to the categorical features

        Returns
        -------
        scores (dict):
            keys (String): name of dynthesizer
            values (pd.DataFrame):
                    ks_statistic (float):
                        KS test statistic.
                    p_value (float):
                        p-value of the test.
        """
        scores = dict()
        scores_before_aggregation = dict()
        for key in self.d_synthetic_data:
            scores_before_aggregation[key] = list()
            for column in self.df_real_data.columns:
                if column in self.categorical_features:
                    # Perform G-square
                    contingency_table = pd.crosstab(
                        self.df_real_data[column], self.d_synthetic_data[key][column]
                    )
                    contingency_table = contingency_table.values
                    chi2_stat, _, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum()
                    k, r = contingency_table.shape
                    cramers_v = np.sqrt(chi2_stat / (n * min(k - 1, r - 1)))
                    scores_before_aggregation[key].append(cramers_v)
                else:
                    # Perform Wasserstein distance
                    wasserstein_dist = wasserstein_distance(
                        self.df_real_data[column], self.d_synthetic_data[key][column]
                    )
                    # Store the Wasserstein distance score in the dictionary
                    scores_before_aggregation[key].append(wasserstein_dist)
        df_scores_before_aggregation = pd.DataFrame(scores_before_aggregation)
        _, _, rankings_avg, _ = friedman_aligned_ranks_test(
            df_scores_before_aggregation
        )
        Ranking = pd.DataFrame([])
        Ranking["Methods"] = scores_before_aggregation.keys()
        Ranking["FAR"] = rankings_avg
        Ranking = Ranking.sort_values(by="FAR", ignore_index=True)
        scores = Ranking.set_index("Methods")["FAR"].to_dict()
        
        if self.verbose:
            print("Wasserstein/Cramers-v test")
            print(50*"-")
            for k, v in scores.items():
                print(f"{k} score: {v:.4f}")
            print("\n")
            
        self.evaluation_metric_scores["wasserstein_cramers_v"] = scores
        return scores

    def novelty_test(self) -> dict[str, float]:
        """
        This function measures whether each row in the synthetic data is novel
        or whether it exactly matches an original row in the real data.

        Returns
        -------
        score (float)
        (best) 0.0: The rows in the synthetic data are all new. There are no matches with the real data.
        (worst) 1.0: All the rows in the synthetic data are copies of rows in the real data.
        """
        # creating the metadata using the Python API
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.df_real_data)
        scores = dict()
        for key in self.d_synthetic_data:
            score = NewRowSynthesis.compute(
                real_data=self.df_real_data,
                synthetic_data=self.d_synthetic_data[key],
                metadata=metadata,
                numerical_match_tolerance=0.01,
                synthetic_sample_size=len(self.d_synthetic_data[key]),
            )
            scores[key] = 1.0 - score
        self.evaluation_metric_scores["novelty"] = scores
        if self.verbose:
            print("Novelty test")
            print(50*"-")
            for k, v in scores.items():
                print(f"{k} score: {v:.4f}")
            print("\n")
        return scores

    def anomaly_detection(
        self, anomaly_threshold: float = 0.000001
    ) -> dict[str, float]:
        """
        This function computes the percentage of anomalies, instances in a dataset that deviate significantly from the norm or expected behavior.

        Parameters
        ----------
        anomaly_threshold: float
            The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
            Default is 0.000001
        Returns
        -------
        score (float)
        (best) 1.0: The rows in the synthetic data are all anomalies
        (worst) 0.0: All the rows in the synthetic data follows the same pattern with real data
        """
        scores = dict()
        for key in self.d_synthetic_data:
            # Instantiate the Isolation Forest model
            model = IsolationForest(
                contamination=anomaly_threshold
            )  # Adjust contamination based on expected anomaly rate
            # Fit the model on the real dataset
            model.fit(self.df_real_data.values)
            # Predict anomalies in the synthetic dataset
            predictions_synthetic = model.predict(self.d_synthetic_data[key].values)
            scores[key] = (
                np.count_nonzero(predictions_synthetic == -1)
                / predictions_synthetic.size
            )
        self.evaluation_metric_scores["anomaly"] = scores
        if self.verbose:
            print("Anomaly detection test")
            print(50*"-")
            for k, v in scores.items():
                print(f"{k} score: {v:.4f}")
            print("\n")
        return scores

    def domain_classifier(self, model, n_folds: int = 5) -> dict[str, float]:
        """
        Trains and evaluates a domain classifier using a HistGradientBoostingClassifier.
        Measures whether the test data, either the real or the synthetic data classified to their corresponding class.

        This function assigns cluster labels to real and synthetic data, concatenates them into
        one dataset, and splits the data into training and testing sets.

        Parameters:
        - self.real_data (pd.DataFrame): DataFrame containing real data.
        - self.synthetic_data (pd.DataFrame): DataFrame containing synthetic data.

        Returns:
            average auc score for each synthesizer
        """
        scores = dict()
        df_real_dataset = self.df_real_data.copy()
        df_real_dataset["cluster"] = 0
        for key in self.d_synthetic_data:
            df_synthetic_dataset = self.d_synthetic_data[key].copy()
            df_synthetic_dataset["cluster"] = 1
            d_data = pd.concat(
                [df_real_dataset, df_synthetic_dataset], ignore_index=True
            )
            X_data = d_data.drop(columns=["cluster"])
            y_data = d_data["cluster"]
            classifier = model
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_results = cross_validate(
                classifier, X_data, y_data, cv=kf, scoring="roc_auc", return_train_score=True
            )
    
            if self.verbose:        
                # Extract AUC scores for training and testing sets
                print(f"Method: {key}")
                print(f"> (Train) AUC score: {100.0*np.mean(cv_results['train_score']):.2f}")
                print(f"> (Test) AUC score: {100.0*np.mean(cv_results['test_score']):.2f}")

            scores[key] = np.mean(100.0*cv_results['test_score'])
            
            
        self.evaluation_metric_scores["classifier"] = scores
        if self.verbose:
            print("\n")
            print("Domain classifier test")
            print(50*"-")
            for k, v in scores.items():
                print(f"{k} score: {v:.2f}")
            print("\n")
        return scores
