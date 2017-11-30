import matplotlib.pyplot as plt

# def f(x):
#     return float(format(float(x), '.3f'))

# def precise(arr):
#     result = list(map(f, arr))
#     print(result)
#     return result

class Plotter:
    @staticmethod
    def plot(predict, real, num=50, print=False):
        if num > len(predict):
            num = len(predict)
        if print:
            print(predict[:num])
            print(real[:num])
        plt.xlabel('day')
        plt.ylabel('fluctuation')
        plt.plot(predict[:num], 'b', label='predict')
        plt.plot(real[:num], 'r', label='real')
        plt.show()