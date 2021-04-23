import numpy as np
import matplotlib.pyplot as plt

XNEW = np.linspace(0, 10, num=501, endpoint=True)


def build_static_graphs():
    plt.gcf().set_size_inches(12, 5)

    for a in [2, 4, 10, 20, 50, 100, 150]:
        # for a in [2, 6, 10, 15, 20, 50, 100, 150, 500, 1000, 1000000000]:
        y = 1 - (a ** (1 - 1/(XNEW*2 + 0.00000001)) / a)
        plt.plot((XNEW*2), y, label='$\\alpha$ = {}'.format(a))

    plt.ylabel('Privacy Value')
    plt.xlabel('Number of People')
    plt.xticks(np.arange(0, 21, step=1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/ap_values.png')
    plt.clf()

    for a in [2, 6, 10, 15, 20, 50, 100, 150, 500, 1000, 1000000000]:
        y = (a ** (1 - 1/(XNEW*2 + 0.00000001)) / a)
        plt.plot((XNEW*2), y, label='$\\alpha$ = {}'.format(a))

    plt.ylabel('Unfamiliarity Value')
    plt.xlabel('Number of Unfamiliar People')
    plt.xticks(np.arange(0, 21, step=1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/au_values.png')
    plt.clf()

    colours = ['xkcd:light red', 'xkcd:red', 'xkcd:dark red', 'xkcd:light green',
               'xkcd:lime green', 'xkcd:green', 'xkcd:light blue', 'xkcd:bright blue', 'xkcd:blue']
    index = 0
    for a in [1.1, 2, 3]:
        for b in np.arange(3, 0, step=-1):
            x = np.concatenate((np.array([0, b]), XNEW + b))
            y = np.concatenate(
                (np.array([1, 1]), 1 - (a ** (1 - 1/(XNEW + 0.00000001)) / a)))
            y = y[x < 10]
            x = x[x < 10]
            plt.plot(x, y, label='$\\alpha_D$ = {}, $\\alpha_d = {}$'.format(
                a, b), color=colours[index % len(colours)], lw=None if index != 5 else 3)
            index += 1

    plt.ylabel('Proximity Value')
    plt.xlabel('Distance (m)')
    plt.xticks(np.arange(0, 11, step=1))
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/ad_values.png')
    plt.clf()


if __name__ == '__main__':
    build_static_graphs()
