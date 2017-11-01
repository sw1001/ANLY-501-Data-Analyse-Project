import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import warnings


def main():
    warnings.filterwarnings("ignore")

    zillow_data = pd.read_csv('../../input/clean/Zillow_Cleaned.csv', index_col=0)
    crime_count_data = pd.read_csv('../../input/clean/crime_counts_CLEANED.csv', index_col=0)
    crime_rates_data = pd.read_csv('../../input/clean/crime_rates_CLEANED.csv', index_col=0)
    earning_info_data = pd.read_csv('../../input/clean/earning_info_CLEANED.csv', index_col=0, encoding='iso-8859-1')
    gdp_info_data = pd.read_csv('../../input/clean/gdp_info_CLEANED.csv', index_col=0)
    graduation_rates_data = pd.read_csv('../../input/clean/graduation_rates_CLEANED.csv', index_col=0)

    zillow_data['latitude'] /= 10e5
    zillow_data['longitude'] /= 10e5

    sample = zillow_data[:1000]
    true_data = zillow_data[:-24700]

    res = stats.ttest_ind(sample['amount'], true_data['amount'])
    print(res)

    early = zillow_data[zillow_data['yearbuilt'] <= 1960]
    late = zillow_data[zillow_data['yearbuilt'] > 1960]
    # print(len(early))  # 15638
    # print(len(late))  # 10081
    print(early['amount'].mean())
    print(late['amount'].mean())

    res = stats.ttest_ind(early['amount'], late['amount'])
    print(res)

    zillow_y_train = zillow_data['amount']
    zillow_x_train = zillow_data.drop(['zpid', 'amount', 'countryid'], axis=1)
    X2 = sm.add_constant(zillow_x_train)
    est = sm.OLS(zillow_y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    svr = SVR(kernel='sigmoid')
    lm = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(zillow_x_train, zillow_y_train)

    svr.fit(X_train, y_train)
    lm.fit(X_train, y_train)

    print(svr.score(X_test, y_test))
    print(lm.score(X_test, y_test))

    zillow_y_train = zillow_data['lotsizeSqFt']
    zillow_x_train = zillow_data[['bathrooms', 'bedrooms']]
    zillow_x_train['rooms'] = zillow_data['bathrooms'] + zillow_data['bedrooms']
    zillow_x_train = zillow_x_train.drop(['bathrooms', 'bedrooms'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(zillow_x_train, zillow_y_train)

    regr = RandomForestRegressor(max_depth=3, random_state=0)
    regr.fit(X_train, y_train)
    print(regr.score(X_test, y_test))

    # Apply a simple xgboost model and do Cross Validata
    zillow_y_train = zillow_data['amount']
    zillow_x_train = zillow_data.drop(['zpid', 'amount', 'countryid'], axis=1)

    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(zillow_x_train, zillow_y_train)

    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
    plt.show()



if __name__ == "__main__":
    main()
