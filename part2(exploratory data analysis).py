'''Analyzing the target variable involves studying its seasonality and trend.
Our aim is to visually understand the patterns and fluctuations in the time series data without heavily relying on statistical techniques such as decomposition.
By graphically examining the data, we can gain insights into the underlying patterns and trends that may exist.
'''

#target Analysis (normality)

mean = np.mean(data.energy.values)
std = np.std(data.energy.values)
skew = skew(data.energy.values)
ex_kurt = kurtosis(data.energy)
print("Skewness: {} \nKurtosis: {}".format(skew, ex_kurt+3))

'''output:
          skewness:-0.2555279252628293
          kurtosis: 2.6052'''

'''In terms of data distribution, negative skewness indicates that the data is not perfectly symmetrical and has a longer left tail. 
  Additionally, the kurtosis value below 3 suggests that the tails of the distribution are slightly thinner compared to a normal distribution.
  This characteristic is known as platykurtic, indicating that the likelihood of encountering extreme values is lower than in a normal distribution.'''


def shapiro_test(data, alpha=0.05):
    stat, pval = shapiro(data)
    print("H0: Data was drawn from a Normal Ditribution")
    if (pval < alpha):
        print("pval {} is lower than significance level: {}, therefore null hypothesis is rejected".format(pval, alpha))
    else:
        print("pval {} is higher than significance level: {}, therefore null hypothesis cannot be rejected".format(pval,
                                                                                                                   alpha))


shapiro_test(data.energy, alpha=0.05)

