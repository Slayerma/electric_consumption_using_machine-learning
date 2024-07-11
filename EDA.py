import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis, shapiro

def plot_energy_demand(data):
    data.plot(title="Energy Demand")
    plt.ylabel("MWh")
    plt.show()

def calculate_statistics(data):
    mean = np.mean(data.energy.values)
    std = np.std(data.energy.values)
    skewness = skew(data.energy.values)
    ex_kurt = kurtosis(data.energy)
    print("Skewness: {} \nKurtosis: {}".format(skewness, ex_kurt + 3))
    return mean, std

def plot_distribution(data, mean, std):
    sns.histplot(data.energy, kde=True)
    plt.title("Target Analysis")
    plt.xticks(rotation=45)
    plt.xlabel("(MWh)")
    plt.axvline(x=mean, color='r', linestyle='-', label=f"\mu: {mean:.2f}")
    plt.axvline(x=mean + 2 * std, color='orange', linestyle='-')
    plt.axvline(x=mean - 2 * std, color='orange', linestyle='-')
    plt.legend()
    plt.show()

def shapiro_test(data, alpha=0.05):
    pval = shapiro(data).pvalue
    print("H0: Data was drawn from a Normal Distribution")
    print(f"p-value {pval} is {'less' if pval < alpha else 'greater'} than significance level: {alpha}")
