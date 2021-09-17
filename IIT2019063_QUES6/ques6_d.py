
# Package imported numpy and pandas and math
import numpy as np
import pandas as pd
import math

# -----------------------------------------------
# dataset csv file is attached within the file...
def My_Predictor():
    def getError(matrix_B, k):
        error = 0
        for i in range(len(matrix_B)):
            error = error + abs(matrix_B[i] - k[i]) / matrix_B[i]
        error = error / len(matrix_B)
        return error * 100


    def kernel(matrix, xi, hyper_tau):
        return   np.exp (-np.sum((xi - matrix) ** 2, axis=1) / (2 * hyper_tau * hyper_tau))


    def LWR(matrix_A, x_i, matrix_B, hyper_tau):
        matrix_A_transpose = np.transpose(matrix_A)

        matrix = kernel(matrix_A, x_i, hyper_tau)
        matrix_transpose = matrix_A_transpose * matrix
        matrix_transposexmatrix_A = np.matmul(matrix_transpose, matrix_A)
        temp = matrix_transposexmatrix_A
        inverse_temp = np.linalg.pinv(matrix_transposexmatrix_A)
        inverse_tempxmatrix_trans = np.matmul(inverse_temp, matrix_transpose)
        temp = inverse_tempxmatrix_trans
        inverse_ATxB = np.matmul(inverse_tempxmatrix_trans, matrix_B)
        inverse_ATxB_T = np.transpose(inverse_ATxB)

        return inverse_ATxB_T.dot(x_i)

    input_data = pd.read_csv('Housing Price data set.csv', usecols=["price", "lotsize", "bedrooms", "bathrms"])
    area = input_data['lotsize']
    bedrooms = input_data['bedrooms']
    bathrooms = input_data['bathrms']
    matrix_B = input_data['price']
    matrix_B = np.array(matrix_B)
    matrix_B = matrix_B.reshape(matrix_B.shape[0], 1)

    # Performing feature scanning on FloorArea
    area_Mean = np.mean(area)
    area_Max = max(area)
    area_Min = min(area)
    area_scaled = []
    for i in area:

        area_scaled.append((i - area_Mean) / (area_Max - area_Min))

    matrix_A = []
    for i in range(len(area)):
         matrix_A.append([1, area_scaled[i], bedrooms[i], bathrooms[i]])

    matrix_A = np.array(matrix_A)

    hyper_tau = 0.00005
    print("Using Locally Weighted Linear Regression for Tau = " + str(hyper_tau))
    pred = []
    for i in range(matrix_A.shape[0]):
        y_pred = LWR(matrix_A, matrix_A[i], matrix_B, hyper_tau)
        pred.append(y_pred)

    print("Mean absolute % error is : " + str(getError(matrix_B, pred)))

    print()

    price = input_data['price']

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
                            temp = temp + coefficient[j] * features_train[i][j]
                        error = error + ((temp - price_train[i]) * features_train[i][idx])

                    return error

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
                                    (learning_rate / m) * (getSlope(  features_train, price_train,coefficient, j)))

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

        def Stochastic_gradientDescent():
            def runCode():
                def getSlopeStochastic( features_train, ActualVal,coefficient, idx):
                    temp = 0
                    for j in range(len(coefficient)):
                        temp = temp + coefficient[j] * features_train[j]

                    return ((temp - ActualVal) * features_train[idx])

                # Using Scaled Stochastic gradient with regularisation.
                print("Using Stochastic gradient with regularisation")

                # different values of tau was tried.

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
                                temp[j] = temp[j] - (learning_rate * (getSlopeStochastic( features_train[i], price_train[i],coefficient, j)))
                            else:
                                temp[j] = (1 - learning_rate * lambda_para / m) * temp[j] - (
                                        learning_rate * (getSlopeStochastic(  features_train[i], price_train[i],coefficient, j)))

                        coefficient = temp.copy()

                print("Final coefficients: ")
                print(coefficient)

                # Finding Mean absolute percentage error.
                error = 0
                for i in range(len(features_test)):
                    predicted = 0
                    for j in range(len(coefficient)):
                        predicted = predicted + coefficient[j] * features_test[i][j]
                    error = error +( abs(predicted - price_test[i]) / price_test[i])

                error = ((error / len(features_test)) * 100)
                print("Mean absolute % error is : " + str(error))
                print()
            runCode()

        def MiniBatch_gradientDescent():
            def runCode():

                # Using Scaled Minibatch gradient with regularisation for batch size = 30
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
