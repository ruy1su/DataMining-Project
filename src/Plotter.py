import matplotlib.pyplot as plt

# def f(x):
#     return float(format(float(x), '.3f'))

# def precise(arr):
#     result = list(map(f, arr))
#     print(result)
#     return result

class Plotter:
    @staticmethod
    def plot(predict, real):
        print(predict[:50])
        print(real[:50])
        # plt.xlabel('day')
        # plt.ylabel('fluctuation')
        # plt.plot(predict[:30], 'b', label='predict')
        # plt.plot(real[:30], 'r', label='real')
        # plt.show()