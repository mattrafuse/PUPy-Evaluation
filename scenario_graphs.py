import numpy as np
from scenarios import SCENARIOS
import scipy.stats as ss
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

from actions import enumerate_actions

XNEW = np.linspace(0, 10, num=501, endpoint=True)

scenario_names = [
    'Coffee Shop Example (Authentication/Device Theft)',
    'Commuter Train (Device Loss)',
    'Dinner Party',
    'Workday',
    'Desk Simulation',
    'Abstract Example'
]

colors = [
    'purple',
    'blue',
    'green',
    'orange'
]

module_colors = [
    'cyan',
    'darkblue',
    'lime'
]

ylabels = [
    'Context Familiarity',
    'Privacy',
    'Unfamiliarity',
    'Proximity',
    'Actions'
]


def build_scenario_graphs():
    plt.gcf().set_size_inches(25, 10)
    for scen_number in range(len(SCENARIOS)):
        print('Building scenario {}'.format(
            scenario_names[scen_number]))
        plt.title(scenario_names[scen_number])

        graphs = SCENARIOS[scen_number].copy()
        actions = graphs.pop(0)

        def subplot(ind, scale='linear'):
            plt.subplot(len(graphs), 1, ind)
            # plt.subplot(len(graphs) + 2, 1, ind)
            plt.yscale(scale)

        subplot(1)
        plt.xticks(np.linspace(0, 10, num=11, endpoint=True))
        plt.yticks(np.linspace(0, 30, num=7, endpoint=True))
        plt.grid(linestyle='--')
        plt.axis((-0.25, 10.25, -0.1, 30.1))
        plt.ylabel('Scenario')

        for p in actions:
            plt.plot(p[0], p[1], '.', color='red')
            plt.annotate(p[2], (p[0] + 0.045, p[1] - 0.01))

        plt.plot([g[0] for g in graphs[1]], [g[1] for g in graphs[1]])
        plt.plot([g[0] for g in graphs[2]], [g[1] for g in graphs[2]])

        functions = []
        for graph_index, graph in enumerate(graphs):
            if graph_index == 1:
                def eq(x, alpha):
                    exponent = 1 - (1 / (np.maximum(x, 0) + .0000001))
                    return 1 - (alpha ** exponent) / alpha
            if graph_index == 2:
                def eq(x, alpha):
                    exponent = 1 - (1 / (np.maximum(x, 0) + .0000001))
                    return (alpha ** exponent) / alpha
            elif graph_index == 0 or graph_index == 3:
                def eq(x, alpha):
                    return x

            middle = interp1d(
                [p[0] for p in graph],
                [p[1] for p in graph],
                kind='linear'
            )

            if graph_index != 0:
                bound = interp1d(
                    [p[0] for p in graph],
                    [(p[2] / 2) for p in graph],
                    kind='linear'
                )
            else:
                bound = interp1d(
                    [p[0] for p in graph],
                    [0 for p in graph],
                    kind='linear'
                )

            alpha = interp1d(
                [p[0] for p in graphs[0]],
                [p[1] for p in graphs[0]],
                kind='linear'
            )

            functions.append((middle, bound, alpha, eq))

        for index, fns in enumerate(functions):
            if index == len(functions) - 1:
                continue

            middle, bound, alpha, eq = fns

            subplot(index + 2)
            plt.xticks(np.linspace(0, 10, num=11, endpoint=True))
            if index == 0:
                plt.axis((-0.25, 10.25, 0, 201))
            else:
                plt.axis((-0.25, 10.25, -0.1, 1.1))
            plt.grid(linestyle='--')
            plt.ylabel(ylabels[index])

            def upper(x):
                return eq(middle(x), alpha(x)) + bound(x)

            def lower(x):
                return eq(middle(x), alpha(x)) - bound(x)

            plt.fill_between(XNEW, lower(XNEW), upper(
                XNEW), color=colors[index], alpha=0.15)
            plt.plot(XNEW, eq(middle(XNEW), alpha(XNEW)),
                     '-', color=colors[index])

        # subplot(len(graphs) + 2)
        # actions = graphs.pop(0)
        # plt.xticks(np.linspace(0, 10, num=11, endpoint=True))
        # plt.yticks([])
        # plt.grid(linestyle='--')
        # plt.axis((-0.25, 10.25, -0.1, 1.1))
        # plt.ylabel('Scenario')

        # plt.annotate('Device Auth', (0.075, 0.7))
        # plt.annotate('Device Theft', (0.075, 0.4))
        # plt.annotate('Device Loss', (0.075, 0.1))

        # activated = [-1, -1, -1]

        # count = 0
        # for i in range(len(XNEW) - 1):
        #     old_state = tuple(
        #         [fns[3](fns[0](XNEW[i]), fns[2](XNEW[i])) for fns in functions])
        #     new_state = tuple(
        #         [fns[3](fns[0](XNEW[i + 1]), fns[2](XNEW[i + 1])) for fns in functions])

        #     result = enumerate_actions(old_state, new_state)

        #     if len(result) > 0:
        #         for res in result:
        #             if type(res[1]) is str and scen_number + 1 != len(SCENARIOS):
        #                 plt.plot(XNEW[i], (res[0] + .5) / 3,
        #                          '.', color='red')
        #                 plt.annotate(
        #                     res[1], (XNEW[i] + 0.075, (res[0] + .5) / 3))
        #                 count += 1
        #             else:
        #                 if res[1] and activated[res[0]] < 0:
        #                     activated[res[0]] = XNEW[i]
        #                 elif not res[1] and activated[res[0]] >= 0:
        #                     plt.fill_between(
        #                         [activated[res[0]], XNEW[i - 1]], 1 - (res[0] / 3), 1 - ((res[0] + 1) / 3), color=module_colors[res[0]], alpha=0.33)

        #                     activated[res[0]] = -1

        # for i, x in enumerate(activated):
        #     if x >= 0:
        #         plt.fill_between(
        #             [activated[i], XNEW[len(XNEW) - 1]], 1 - (i / 3), 1 - ((i + 1) / 3), color=module_colors[i], alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            'plots/scenario={}.pdf'.format(scen_number, alpha))
        plt.clf()


if __name__ == '__main__':
    build_scenario_graphs()
