import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(predict, real):
        # print(predict[:10])
        # print(real[:10])
        plt.xlabel('day i')
        plt.ylabel('fluctuation')
        plt.plot(predict[:30], 'b', label='predict')
        plt.plot(real[:30], 'r', label='real')
        plt.show()