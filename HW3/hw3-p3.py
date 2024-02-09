import numpy as np
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
# from pmdarima.arima import ndiffs
# from pmdarima.arima.utils import nsdiffs
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

if __name__ == '__main__':
    ### Load data ###
    data_train = pd.read_csv('hw3_Data2/train.csv')
    data_test = pd.read_csv('hw3_Data2/test.csv')

    train = data_train['Close'].to_numpy()
    test = data_test['Close'].to_numpy()


    ### Check the stationarity of the time series ###
    # fig, axes = plt.subplots(3, figsize=(10, 8))
    # axes[0].plot(data_train.Close)
    # axes[0].set_title('Original Series')
    # axes[0].axes.xaxis.set_visible(False)
    
    # axes[1].plot(data_train.Close.diff().dropna())
    # axes[1].set_title('1st Order Differencing')
    # axes[1].axes.xaxis.set_visible(False)

    # axes[2].plot(data_train['Date'][:223], data_train.Close.diff().diff().dropna())
    # axes[2].xaxis.set_ticks(np.arange(0, 223, 42))
    # axes[2].set_title('2nd Order Differencing')
    # plt.savefig('./images/hw3-3-0_diff.png')
    # plt.show()


    ### Find d and D ###
    # n_diffs = max(ndiffs(train, alpha=0.05, test='adf', max_d=6), ndiffs(train, alpha=0.05, test='kpss', max_d=6))
    # print(f'By using adf and kpss methods,\td = {n_diffs}')
    print(f'By using adf and kpss methods,\td = 1')

    # n_sdiffs = max(nsdiffs(train, m=30, max_D=12, test='ch'), nsdiffs(train, m=30, max_D=12, test='ocsb'))
    # print(f'By using ch and ocsb methods,\tD = {n_sdiffs}')
    print(f'By using ch and ocsb methods,\tD = 0')


    ### Plot ACF and PACF ###
    # train_diff1 = data_train.Close.diff().dropna()
    # plot_acf(np.array(train_diff1), lags=50)
    # plt.ylim(-1.1, 1.1)
    # plt.savefig('./images/hw3-3-1_acf.png')
    # plt.show()
    # plot_pacf(np.array(train_diff1), lags=50, method='ywm')
    # plt.ylim(-1.1, 1.1)
    # plt.savefig('./images/hw3-3-2_pacf.png')
    # plt.show()
    print(f'By plotting ACF and PACF,\tp = 0, q = 0')


    ### Find the optimal order ###
    # model = pm.auto_arima(train, start_p=0, max_p=0, d=1, max_d=2, start_q=0, max_q=0, m=30,
    #                       seasonal=True, start_P=0, max_P=2, D=0, max_D=2, start_Q=0, max_Q=2,  
    #                       trace=True, error_action='ignore', suppress_warnings=True, stepwise=True) 
    # print(model.summary())


    ### Execute the ARIMA model ###
    seasonal = [(1, 0, 1, 30), (2, 1, 0, 30), (1, 0, 1, 30)]
    iterable = [False, False, True]
    for i in range(3):
        ### Train the model ###
        model = pm.arima.ARIMA(order=(0, 1, 0), seasonal_order=seasonal[i], suppress_warnings=True)
        # print(model.get_params())
        model.fit(train)


        ### Create predictions for the future, evaluate on test ###
        if iterable[i] == False:
            prediction, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
        else:
            prediction = np.zeros_like(test)
            conf_int = np.zeros((len(test), 2))
            for idx, sample in enumerate(test):
                prediction[idx], conf_int[idx, :] = model.predict(n_periods=1, return_conf_int=True)
                model.update(sample)


        ### Compute the mean squared error ###
        mse = ((test - prediction) ** 2).mean()
        print(f'Test{i} MSE: {mse:.3f}')


        ### Plot the results ###
        # x_axis = np.arange(train.shape[0] + test.shape[0])
        x_axis = data_train['Date'].to_list() + data_test['Date'].to_list()

        plt.figure(figsize=(12, 8))
        plt.title('ARIMA')
        plt.plot(x_axis[:train.shape[0]], train, alpha=0.75)
        plt.plot(x_axis[train.shape[0]:], test, alpha=0.75)  # Test data
        plt.plot(x_axis[train.shape[0]:], prediction, alpha=0.75)  # Forecasts
        plt.xlabel('Date')
        plt.ylabel('Close value')
        plt.legend(['train', 'test', 'prediction'])
        plt.fill_between(x_axis[train.shape[0]:], conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
        plt.xticks(x_axis[::21], rotation=30)
        # plt.savefig(f'./images/hw3-3-{i + 3}_result{i}.png')
        plt.show()