'''ELECTRICITY CONSUMPTION USING MACHINE LEARNING'''

'''About the Dataset
This dataset is a daily time series of electricity demand, generation, and prices in Spain from 2014 to 2018. It is gathered from ESIOS, a website managed by REE (Red Electrica Española), which is the Spanish TSO (Transmission System Operator)

A TSO's main function is to operate the electrical system and to invest in new transmission (high voltage) infrastructure.

(https://www.ree.es/en/about-us/business-activities/electricity-business-in-Spain)'''

'''Content
Original values are kept, so some names in Spanish are shown. The column name describes each time series, so I provide a description of each name:

Demanda programada PBF total (MWh): Schedulled Total Demand

Demanda real (MW): Actual demanded power

Energía asignada en Mercado SPOT Diario España (MWh): Energy traded in daily spot Spanish market (OMIE)

Energía asignada en Mercado SPOT Diario Francia (MWh): Energy traded in daily spot French market

Generación programada PBF Carbón (MWh): Schedulled Coal electricity generation

Generación programada PBF Ciclo combinado (MWh): Schedulled Combined Cycle electricity generation

Generación programada PBF Eólica: (MWh): Schedulled Wind electricity generation

Generación programada PBF Gas Natural Cogeneración (MWh): Schedulled Natural Gas electricity Co-generation

Generación programada PBF Nuclear (MWh): Schedulled Nuclear electricity generation

Generación programada PBF Solar fotovoltaica (MWh): Schedulled Photovoltaic electricity generation

Generación programada PBF Turbinación bombeo (MWh): Schedulled Reversible-Hydro electricity generation

Generación programada PBF UGH + no UGH (MWh): Schedulled Total Hydroelectricity generation

Generación programada PBF total (MWh): Schedulled Total electricity generation

Precio mercado SPOT Diario ESP (€/MWh): Daily spot Spain market price

Precio mercado SPOT Diario FRA (€/MWh): Daily spot France market price

Precio mercado SPOT Diario POR (€/MWh): Daily spot Portugal market price

Rentas de congestión mecanismos implícitos diario Francia exportación (€/MWh): Daily spot export from France price

Rentas de congestión mecanismos implícitos diario Francia importación (€/MWh): Daily spot import to France price

Rentas de congestión mecanismos implícitos diario Portugal exportación (€/MWh): Daily spot export from Portugal price

Rentas de congestión mecanismos implícitos diario Portugal importación (€/MWh): Daily spot import to Portugal price'''


<b>EDA(Exploratory Data Analysis)</b>
Analyzing the target variable involves studying its seasonality and trend.
Our aim is to visually understand the patterns and fluctuations in the time series data without heavily relying on statistical techniques such as decomposition. 
By graphically examining the data, we can gain insights into the underlying patterns and trends that may exist.
'''

<b> Target Analysis(Normality) </b>
mean = np.mean(data.energy.values)  
std = np.std(data.energy.values)  
skew = skew(data.energy.values)  
ex_kurt = kurtosis(data.energy)  
print("Skewness: {} \nKurtosis: {}".format(skew, ex_kurt+3))  

In terms of data distribution, negative skewness indicates that the data is not perfectly symmetrical and has a longer left tail. 
Additionally, the kurtosis value below 3 suggests that the tails of the distribution are slightly thinner compared to a normal distribution. 
This characteristic is known as platykurtic, indicating that the likelihood of encountering extreme values is lower than in a normal distribution.

<b> Feature Engineering</b>
The current challenge lies in developing automated features that can effectively handle seasonality, trend, and changes in volatility. 
These features should be able to adapt to the varying patterns and fluctuations observed in the data.
Standardizing the data is a necessary step to enable the application of models that are sensitive to scale, such as neural networks or support vector machines (SVM). By standardizing the data, we ensure that the distribution shape remains unchanged while only altering the first and second moments, namely the mean and standard deviation.
This process allows for more accurate and effective modeling of the data using these particular machine learning algorithms.

<b>Model Building</b>
In this step, we have built two candidate models using a convenient feature in Scikit-Learn called MultiOutput Regression.
This feature allows us to efficiently and automatically fit models that can predict multiple target variables simultaneously. By leveraging this framework, we can train our models to predict several target variables in a streamlined manner. This not only simplifies the modeling process but also enables us to evaluate the models' performance across multiple targets effectively.

First, we will fit a baseline model using linear regression and compare it to a more advanced model, such as Random Forest. 
The linear regression model does not require extensive hyperparameter tuning and provides a solid foundation for our analysis. However, there are several considerations to keep in mind:

<b> Non-Normal Distribution and Varied Variance:</b> The target variable does not follow a perfect normal distribution and exhibits varying levels of variance. 
    This can affect the assumptions of linear regression, which assumes normality and constant variance. 
    We need to be cautious of potential deviations from these assumptions.
<b> Multicollinearity Among Predictors:</b> There is a high degree of multicollinearity among the predictor variables, meaning that some predictors are highly correlated with each other. 
   This can introduce challenges in interpreting the individual effects of these predictors on the target variable and may impact the model's performance.
<b> Non-Independence of Observations:</b> The observations in our dataset may not be independent, which violates one of the key assumptions of linear regression.
   Non-independence can arise from various factors, such as temporal dependencies or clustering within the data. 
   We need to consider this when interpreting the model results and evaluating its accuracy.
On the other hand, an advanced model such as Random Forest requires careful hyperparameter tuning to achieve optimal performance. Typically, this is done using techniques like GridSearch and Cross Validation (CV). However, using traditional CV methods with time series data poses challenges. This is because the data should not be shuffled as it follows a specific time structure.

Fortunately, Scikit-Learn provides a helpful solution called TimeSeries Split. This technique allows us to perform GridSearch in a time-aware manner by preserving the temporal order of the data. It splits the data into sequential time-based folds, ensuring that each fold respects the chronological order of the observations.

By using TimeSeries Split, we can iteratively train and evaluate our Random Forest model with different combinations of hyperparameters. This approach enables us to find the best set of hyperparameters that maximizes the model's performance on unseen future data points.

Applying hyperparameter tuning in a time-aware manner is essential for time series data, as it ensures that our model's performance is more realistic and reliable.
By leveraging the TimeSeries Split functionality in Scikit-Learn, we can effectively optimize our Random Forest model without violating the temporal structure of the data
.
<b> Splitting Data </b>
To ensure an unbiased evaluation of our model's performance and conduct thorough residual analysis, we reserve the data points from the year 2018 as a separate holdout dataset. T
this means that we keep this data untouched during the model development process.

<b>Train a Random Forest with Time Series Split to tune Hyperparameters </b>
In this particular example, we illustrate the use of the TimeSeriesSplit framework.
With this approach, each fold of the data is constructed in such a way that the training data is closer to the beginning of the forecasting period.

<b> refer model assessment </b>

<b>Forecasting</b>
<b>Multi-period ahead model building</b>
Once we determine the optimal set of hyperparameters, we can train a new instance of the Random Forest model using the most recent and relevant data. 
Typically, it is recommended to have at least two years of data to generate a long-term daily forecast. 
Let's proceed with retraining a collection of Random Forest models using the MultiOutput Regression feature

<b> Future Aspects of Machine Learning </b>
Machine learning holds immense potential for the energy industry. 
By analyzing vast historical data, including electricity usage, weather patterns, and seasonal variations, machine learning algorithms can provide accurate predictions. 
Challenges such as complex relationships between variables are being addressed with advanced techniques.
The future of this field looks promising, with improved accuracy, integration of IoT and smart grid data, and real-time predictive analytics. 
This will enable efficient energy distribution, demand-side response, and seamless integration of renewable energy sources.
Moreover, machine learning will support predictive maintenance for energy infrastructure and foster energy conservation and sustainability. 
The collaboration between AI and human expertise will be essential, and transparent AI models will build trust and accountability. 
Overall, machine learning is set to transform the energy sector and pave the way for a more sustainable and efficient energy ecosystem.

<b> Conclusion </b>
