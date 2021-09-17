# This code demonstrates the difference between a regularised hypothesis and a non regularised hypothesis
# using Batch Gradient Descent, Stochastic Gradient Descent and Minibatch Gradient Descent.

# Package imported numpy and pandas and math
import numpy as np
import pandas as pd
import math

# -----------------------------------------------
# dataset csv file is attached within the file...

input_data = pd.read_csv("Housing Price data set.csv")

def My_Predictor():
    price = input_data['price']
    area = input_data['lotsize']
    bedrooms = input_data['bedrooms']
    bathrooms = input_data['bathrms']

    # Performing feature scanning on FloorArea
    area_MeanValue = np.mean(area)
    area_MaxValue = max(area)
    area_MinValue = min(area)
    area_scaled = []
    for i in area:
        area_scaled.append((i - area_MeanValue) / (area_MaxValue - area_MinValue))

    # segmenting the features
    features_train = []
    for i in range(383):
        features_train.append([1, area_scaled[i], bedrooms[i], bathrooms[i]])
    price_train = price[:383]

    price_test = []
    features_test = []
    for i in range(383, len(price)):
        features_test.append([1, area_scaled[i], bedrooms[i], bathrooms[i]])
        price_test.append(price[i])
    m = len(features_train)

    def run_predictors():

        def Batch_gradientDescent():
            def runCode():
                def getSlope(features_train, price_train, coefficient, idx):
                    error = 0
                    for i in range(len(features_train)):
                        temp = 0

                        for j in range(len(coefficient)):
                            temp = temp + features_train[i][j] * coefficient[j]
                        error = error + ((temp - price_train[i]) * features_train[i][idx])
                    return error

                # Using scaled batch gradient without regularisation
                print("Using scaled batch gradient without regularisation")
                learning_rate = 0.001
                m = len(features_train)

                coefficient = [0, 0, 0, 0]
                print("Initial coefficients: ")
                print(coefficient)
                for i in range(5000):
                    temp = coefficient.copy()
                    for j in range(len(coefficient)):
                        temp[j] = temp[j] - ((learning_rate / m) * (getSlope(features_train, price_train, coefficient, j)))
                    coefficient = temp.copy()

                print("Final coefficients: ")
                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 90
                print("Mean absolute percentage error is : " + str(error))
                print()

                # Using scaled batch gradient with regularisation
                print("Using scaled batch gradient with regularisation")
                learning_rate = 0.001
                lambda_para = -51
                coefficient = [0, 0, 0, 0]
                print("Initial coefficients: ")
                print(coefficient)
                for x in range(5000):
                    temp = coefficient.copy()
                    for j in range(len(coefficient)):
                        if (j == 0):
                            temp[j] = temp[j] - ((learning_rate / m) * (getSlope(features_train, price_train, coefficient, j)))
                        else:
                            temp[j] = (1 - learning_rate * lambda_para / m) * temp[j] - (
                                    (learning_rate / m) * (getSlope(features_train, price_train, coefficient, j)))
                    coefficient = temp.copy()

                print("Final coefficients: ")
                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + features_test[i][j] * coefficient[j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 100

                print("Mean absolute % error is : " + str(error))
                print()

            runCode()


        def Stochastic_gradientDescent():
            def runCode():
                def getSlopeStochastic(features_train, ActualVal, coefficient, idx):
                    itr = np.longdouble
                    itr = 0
                    for j in range(len(coefficient)):
                        itr = itr + coefficient[j] * features_train[j]
                    return (itr - ActualVal) * features_train[idx]

                # Using Scaled Stochastic gradient without regularisation.
                print("Using Stochastic gradient without regularisation")

                learning_rate = 0.005
                coefficient = [0, 0, 0, 0]
                print("Initial coefficients: ")
                print(coefficient)

                for iter in range(10):
                    for i in range(len(price_train)):
                        temp = coefficient.copy()
                        for j in range(4):
                            temp[j] = temp[j] - (
                                    learning_rate * (getSlopeStochastic(features_train[i], price_train[i], coefficient, j)))
                        coefficient = temp.copy()

                print("Final coefficients: ")
                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 100
                print("Mean absolute % error is : " + str(error))
                print()

                # Using Scaled Stochastic gradient with regularisation.
                print("Using Stochastic gradient with regularisation")

                learning_rate = 0.005
                lambda_para = 200
                coefficient = [0, 0, 0, 0]
                print("Initial coefficients: ")
                print(coefficient)

                for iter in range(10):
                    for i in range(len(price_train)):
                        temp = coefficient.copy()
                        for j in range(4):
                            if j == 0:
                                temp[j] = temp[j] - (
                                        learning_rate * (getSlopeStochastic(features_train[i], price_train[i], coefficient, j)))
                            else:
                                temp[j] = (1 - learning_rate * lambda_para) * temp[j] - (
                                        learning_rate * (getSlopeStochastic(features_train[i], price_train[i], coefficient, j)))
                        coefficient = temp.copy()

                print("Final coefficients: ")
                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 100
                print("Mean absolute % error is : " + str(error))
                print()
            runCode()


        def MiniBatch_gradientDescent():
            def runCode():
                # Using Scaled Minibatch gradient without regularisation for batch size = 30
                print("Using Scaled Minibatch gradient without regularisation for batch size = 30")

                batch_size = 30;
                learning_rate = 0.002
                coefficient = [0, 0, 0, 0]
                batches = math.ceil(len(price_train) / batch_size)
                equally_divided = False
                if (len(price_train) % batch_size == 0):
                    equally_divided = True;

                for x in range(30):
                    for batch in range(batches):
                        sum = [0, 0, 0, 0]
                        for j in range(len(coefficient)):
                            for i in range(batch_size):
                                if (batch * batch_size + i == len(features_train)):
                                    break
                                predicted_val = 0.0
                                for k in range(len(coefficient)):
                                    predicted_val += coefficient[k] * features_train[batch * batch_size + i][k]
                                predicted_val -= price_train[batch * batch_size + i]
                                predicted_val *= features_train[batch * batch_size + i][j]
                                sum[j] += predicted_val;

                        if (not equally_divided and batch == batches - 1):
                            for j in range(len(sum)):
                                coefficient[j] -= (sum[j] / (len(price_train) % batch_size)) * learning_rate
                        else:
                            for j in range(len(sum)):
                                coefficient[j] -= (sum[j] / batch_size) * learning_rate
                print("Final coefficients: ")

                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 100
                print("Mean absolute % error is : " + str(error))
                print()

                # Using Scaled Minibatch gradient with regularisation for batch size = 20
                print("Using Scaled Minibatch gradient with regularisation for batch size = 30")

                batch_size = 30;
                learning_rate = 0.002
                lambda_para = -355
                coefficient = [0, 0, 0, 0]
                batches = math.ceil(len(price_train) / batch_size)
                equally_divided = False
                if (len(price_train) % batch_size == 0):
                    equally_divided = True;

                for x in range(30):
                    for batch in range(batches):
                        sum = [0, 0, 0, 0]
                        for j in range(len(coefficient)):
                            for i in range(batch_size):
                                if (batch * batch_size + i == len(features_train)):
                                    break
                                predicted_val = 0.0
                                for y in range(len(coefficient)):
                                    predicted_val += coefficient[y] * features_train[batch * batch_size + i][y]
                                predicted_val -= price_train[batch * batch_size + i]
                                predicted_val *= features_train[batch * batch_size + i][j]
                                sum[j] += predicted_val;

                        if (not equally_divided and batch == batches - 1):
                            for j in range(len(sum)):
                                if j == 0:
                                    coefficient[j] -= (sum[j] / (len(price_train) % batch_size)) * learning_rate
                                else:
                                    coefficient[j] = (1 - learning_rate * lambda_para / m) * coefficient[j] - (
                                            sum[j] / (len(price_train) % batch_size)) * learning_rate
                        else:
                            for j in range(len(sum)):
                                if j == 0:
                                    coefficient[j] -= (sum[j] / batch_size) * learning_rate
                                else:
                                    coefficient[j] = (1 - learning_rate * lambda_para / m) * coefficient[j] - (
                                            sum[j] / batch_size) * learning_rate
                print("Final coefficients: ")

                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error += abs(predicted - price_test[i]) / price_test[i]
                error = (error / len(features_test)) * 100
                print("Mean absolute % error is : " + str(error))
                print()
            runCode()

        print("Predictor Using Batch Gradient Descent")
        Batch_gradientDescent()

        print("Predictor Using Stochastic Gradient Descent")
        Stochastic_gradientDescent()

        print("Predictor Using MiniBatch Gradient Descent")
        MiniBatch_gradientDescent()

    run_predictors()

My_Predictor()