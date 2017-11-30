import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(predict, real):
        plt.xlabel('day i')
        plt.ylabel('fluctuation')
        plt.plot(predict, 'b', label='predict')
        plt.plot(real, 'r', label='real')
        plt.show()