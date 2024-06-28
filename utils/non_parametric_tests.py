import numpy as np
import scipy as sp
import scipy.stats as st
import pandas as pd


def friedman_aligned_ranks_test(df: pd.DataFrame)->(float, float, list, list):
    """
    Performs a Friedman aligned ranks ranking test.
    Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.
    The difference with a friedman test is that it uses the median of each group to construct the ranking, which is useful when the number of samples is low.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing data for analysis.
    Returns
    -------
    Chi2-value : float
        The computed Chi2-value of the test.
    p-value : float
        The associated p-value from the Chi2-distribution.
    rankings : array_like
        The ranking for each group.
    pivots : array_like
        The pivotal quantities for each group.

    References
    -----------------------------------------------
     J.L. Hodges, E.L. Lehmann, Ranks methods for combination of independent experiments in analysis of variance, Annals of Mathematical Statistics 33 (1962) 482–497.
    """
    # Convert dataframe to matrix
    M = df.to_numpy()

    # Get number of methods and number of problems
    n, k = M.shape[0], M.shape[1]
    if k < 2:
        raise ValueError("FAR and Finner tests cannot be applied for less than 2 tests")
    if n < 3:
        raise ValueError(
            "FAR and Finner tests cannot be applied for less than 3 methods/models"
        )
    aligned_observations = []
    for i in range(n):
        aligned_observations.extend(M[i, :] - sp.mean(M[i, :]))

    aligned_observations_sort = sorted(aligned_observations)

    aligned_ranks = []
    for i in range(n):
        row = []
        for j in range(k):
            v = aligned_observations[i * k + j]
            row.append(
                aligned_observations_sort.index(v)
                + 1
                + (aligned_observations_sort.count(v) - 1) / 2.0
            )
        aligned_ranks.append(row)

    rankings_avg = [sp.mean([case[j] for case in aligned_ranks]) for j in range(k)]
    rankings_cmp = [r / sp.sqrt(k * (n * k + 1) / 6.0) for r in rankings_avg]

    r_i = [np.sum(case) for case in aligned_ranks]
    r_j = [np.sum([case[j] for case in aligned_ranks]) for j in range(k)]
    T = (
        (k - 1)
        * (sp.sum(v**2 for v in r_j) - (k * n**2 / 4.0) * (k * n + 1) ** 2)
        / float(
            ((k * n * (k * n + 1) * (2 * k * n + 1)) / 6.0)
            - (1.0 / float(k)) * sp.sum(v**2 for v in r_i)
        )
    )

    p_value = 1 - st.chi2.cdf(T, k - 1)
    return T, p_value, rankings_avg, rankings_cmp


def finner_test(ranks, control=None) -> (list, list, list, list):
    """
    Performs a Finner post-hoc test using the pivot quantities obtained by a ranking test.
    Tests the hypothesis that the ranking of the control method is different to each of the other methods.

    Parameters
    ----------
    ranks : dictionary_like
        A dictionary with format 'groupname': 'pivotal quantity'
    control : string optional
        The name of the control method, default the group with minimum ranking

    Returns
    ----------
    Comparions : array-like
        Strings identifier of each comparison with format 'group_i' vs 'group_j'
    Z-values : array-like
        The computed Z-value statistic for each comparison.
    p-values : array-like
        The associated p-value from the Z-distribution wich depends on the index of the comparison
    Adjusted p-values : array-like
        The associated adjusted p-values wich can be compared with a significance level

    References
    -----------------------------------------------
    H. Finner, On a monotonicity problem in step-down multiple test procedures, Journal of the American Statistical Association 88 (1993) 920–923.
    """
    k = len(ranks)
    values = list(ranks.values())
    keys = list(ranks.keys())

    if not control:
        control_i = values.index(min(values))
    else:
        control_i = keys.index(control)

    comparisons = [
        keys[control_i] + " vs " + keys[i] for i in range(k) if i != control_i
    ]
    z_values = [abs(values[control_i] - values[i]) for i in range(k) if i != control_i]
    p_values = [2 * (1 - st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(
        list, zip(*sorted(zip(p_values, z_values, comparisons), key=lambda t: t[0]))
    )
    adj_p_values = [
        min(
            max(
                1 - (1 - p_values[j]) ** ((k - 1) / float(j + 1)) for j in range(i + 1)
            ),
            1,
        )
        for i in range(k - 1)
    ]

    return comparisons, z_values, p_values, adj_p_values


def statistical_analysis(df: pd.DataFrame, verbose:bool=True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Perform statistical analysis on a DataFrame using Friedman Aligned Ranks Test and Finner post-hoc test.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data for analysis.
    - verbose(bool): Flag for presenting results

    Returns:
    tuple: A tuple containing three DataFrames - Ranking, Finner and Li tests.

    The Ranking DataFrame contains the methods, their FAR (Friedman Aligned Ranks), and is sorted based on FAR, adjusted p-values, and whether the null hypothesis is rejected.

    Example:
    Ranking = statistical_analysis(data_frame)
    """
    T, p_value, rankings_avg, _ = friedman_aligned_ranks_test(df)

    if verbose:
        # Friedman Aligned Ranking
        print("[INFO] H0: {All methods exhibited similar results with no statistical differences}")
        if p_value < 0.05:
            print(f"[INFO] FAR: {T:.3f} (p-value: {p_value:.5f} - H0 is rejected)")
        else:
            print(f"[INFO] FAR: {T:.3f} (p-value: {p_value:.5f}) - H0 is failed to be rejected)")

    Ranking = pd.DataFrame([])
    Ranking["Methods"] = df.columns
    Ranking["FAR"] = rankings_avg
    # Sorting based on FAR score
    Ranking = Ranking.sort_values(by="FAR", ignore_index=True)
    d = {Ranking["Methods"][i]: Ranking["FAR"][i] for i in range(Ranking.shape[0])}

    # Finner post-hoc test
    _, _, _, adj_p_values = finner_test(d)
    adj_p_values.reverse()
    
    Ranking["APV"] = ["-"] + adj_p_values
    Ranking["Null hypothesis"] = Ranking["APV"].apply(
        lambda x: "-" if x == "-" else "Rejected" if x < 0.05 else "Failed to reject"
    )
    
    return Ranking
