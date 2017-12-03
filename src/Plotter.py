import matplotlib.pyplot as plt
import utils

# def f(x):
#     return float(format(float(x), '.3f'))

# def precise(arr):
#     result = list(map(f, arr))
#     print(result)
#     return result

class Plotter:
    @staticmethod
    def plot(predict, real, num=50, print=False, classification=False):
        if num > len(predict):
            num = len(predict)
        if print:
            print(predict[:num])
            print(real[:num])
        if classification:
            predict = plt.plot(utils.sign_list(predict[:num]), 'bo', label='predict')
            real = plt.plot(utils.sign_list(real[:num]), 'ro', label='real')
        else:
            plt.plot(predict[:num], 'b', label='predict')
            plt.plot(real[:num], 'r', label='real')

        plt.xlabel('day')
        plt.ylabel('fluctuation')
        plt.legend(loc='best')
        plt.show()