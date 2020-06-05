from scipy import stats


def one_sample_one_tailed(sample_data, popmean, alternative="greater"):
    """Performs a one-tailed one sample t-test."""

    t, p = stats.ttest_1samp(sample_data, popmean)
    alpha = 0.05

    print("t:", t)
    print("p:", p)

    if alternative == "greater" and (p / 2 < alpha) and t > 0:
        print("Reject Null Hypothesis for greater-than test")

    if alternative == "less" and (p / 2 < alpha) and t < 0:
        print("Reject Null Hypothesis for less-thane test")


if __name__ == "__main__":
    sample_data = [
        0.833,
        0.274,
        0.353,
        0.055,
        0.368,
        0.15,
        0.0,
        0.631,
        0.222,
        0.294,
        0.2,
        0.158,
    ]

    one_sample_one_tailed(sample_data, 0.2)
    one_sample_one_tailed(sample_data, 0.353)
    one_sample_one_tailed(sample_data, 0.222)
