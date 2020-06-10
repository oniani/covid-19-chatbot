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
        0.750,
        0.250,
        0.300,
        0.050,
        0.350,
        0.150,
        0.000,
        0.000,
        0.200,
        0.294,
        0.200,
        0.150,
    ]

    one_sample_one_tailed(sample_data, 0.200)
    one_sample_one_tailed(sample_data, 0.300)
    one_sample_one_tailed(sample_data, 0.200)
