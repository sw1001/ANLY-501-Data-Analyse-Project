import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


price_earn = pd.read_csv('../p3_visualization/plot-1/price_earning.csv')


def main():
    zillow_data = pd.read_csv('Zillow_Cleaned_3.csv')

    zillow_data['latitude'] /= 10e5
    zillow_data['longitude'] /= 10e5

    # Feature Engineering

    zillow_y_train = zillow_data['amount']
    zillow_x_train = zillow_data.drop(['zpid', 'amount', 'counties', 'cityid', 'zipcode', 'latitude', 'longitude'], axis=1)

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
                       verbose_eval=10, show_stdv=False)
    num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
    plt.show()
    # plt.savefig('update_2_feature_importance')


if __name__ == "__main__":
    main()
