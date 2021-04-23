from os import path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

from actions import enumerate_actions
from collections import Counter
from multiprocessing import Pool
import os

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
    'Scenario',
    'Context Familiarity',
    'Privacy',
    'Unfamiliarity',
    'Proximity',
    'Modules'
]


def build_haggle_graphs():
    plt.gcf().set_size_inches(25, 10)
    students = list(range(1, 37))

    print('Generating stats')
    with Pool(min(os.cpu_count() - 2, len(students))) as p:
        p.starmap(build_graph_for_student, [
                  (student, False) for student in students])


def build_graph_for_student(student, verbose=True):
    print(f'  Building graph for student {student}')
    data = []
    folder = 'datasets/haggle/imote-traces-cambridge/SR-10mins-Students'
    with open(path.join(folder, '{}.dat'.format(student))) as f:
        tmp_data = f.read().strip()
        tmp_data = tmp_data.split('\n')

        for i in range(len(tmp_data)):
            tmp_data[i] = tmp_data[i].split(' ')

            for j in range(len(tmp_data[i])):
                tmp_data[i][j] = int(tmp_data[i][j])

        data = tmp_data

    top_seen = [x[0] for x in Counter([x[0] for x in data]).most_common(3)]

    timestamps = sorted(set([x[1] for x in data]).union(
        set([x[2] for x in data])))

    random.seed(min(timestamps))
    DEFAULT_ALPHA = 3
    current_alpha = DEFAULT_ALPHA

    privacy = []
    unfamiliarity = []
    proximity = []
    alpha = []
    privacy_val = 0
    unfamiliarity_val = 0
    to_set_alpha = DEFAULT_ALPHA
    to_remove_privacy = 0
    to_remove_unfamiliarity = 0
    for timestamp in timestamps:
        for entry in data:
            if timestamp == entry[1]:
                privacy_val += 1

                if entry[1] == entry[2]:
                    to_remove_privacy += 1

                if entry[0] in top_seen:
                    privacy_val -= 1

                    to_set_alpha = current_alpha = 150

                    if entry[1] == entry[2]:
                        to_remove_privacy -= 1
                        to_set_alpha = DEFAULT_ALPHA
                elif entry[0] > 54:
                    unfamiliarity_val += 1

                    if entry[1] == entry[2]:
                        to_remove_unfamiliarity += 1
                elif entry[0] > 36:
                    privacy_val -= 1
                    to_set_alpha = current_alpha = 25

                    if entry[1] == entry[2]:
                        to_remove_privacy -= 1
                        to_set_alpha = DEFAULT_ALPHA
            elif timestamp == entry[2]:
                privacy_val -= 1

                if entry[0] in top_seen:
                    privacy_val += 1
                    current_alpha = DEFAULT_ALPHA
                    to_set_alpha = DEFAULT_ALPHA
                elif entry[0] > 54:
                    unfamiliarity_val -= 1
                elif entry[0] > 36:
                    privacy_val += 1
                    current_alpha = DEFAULT_ALPHA
                    to_set_alpha = DEFAULT_ALPHA

        exponent = 1 - (1 / (np.maximum(privacy_val, 0) + .0000001))
        current_result = 1 - (current_alpha ** exponent) / current_alpha
        privacy.append(current_result)

        exponent = 1 - (1 / (np.maximum(unfamiliarity_val, 0) + .0000001))
        current_result = (current_alpha ** exponent) / current_alpha
        unfamiliarity.append(current_result)

        if len(alpha) > 1:
            alpha.append(current_alpha * 0.3 + alpha[len(alpha) - 1] * 0.7)
        else:
            alpha.append(current_alpha)

        privacy_val -= to_remove_privacy
        unfamiliarity_val -= to_remove_unfamiliarity
        current_alpha = to_set_alpha

        to_remove_privacy = 0
        to_remove_unfamiliarity = 0

        proximity.append(1 - (random.randint(0, 100) / 100) ** 4)

    plt.title('Person ' + str(student))

    def subplot(ind, scale='linear'):
        ax = plt.subplot(5, 1, ind)
        plt.yscale(scale)
        ax.set_xlim((min(timestamps) - 6 * 60 * 60,
                     max(timestamps) + 6 * 60 * 60))
        return ax

    functions = []

    for index, values in enumerate([alpha, privacy, unfamiliarity, proximity]):

        ax = subplot(index + 1)
        ticks = np.linspace(min(timestamps),
                            max(timestamps), num=10, endpoint=True)
        date_ticks = [str(datetime.fromtimestamp(round(x / 60) * 60))
                      for x in ticks]
        ax.set_xticklabels(date_ticks)
        plt.xticks([round(x / 60) * 60 for x in ticks])
        # if index == 0:
        #     plt.xlim([-0.25, 10.25])
        #     plt.yscale('log')
        # else:
        #     plt.axis((-0.25, 10.25, -0.1, 1.1))
        # plt.grid(linestyle='--')
        plt.ylabel(ylabels[index + 1])
        if index > 0:
            ax.set_ylim((-0.1, 1.1))

        plt.plot(timestamps, values, '-', color=colors[index])

    for entry in data:
        if entry[0] < 54 and entry[0] > 36 and entry[0] not in [54, 51]:
            if entry[1] == entry[2]:
                plt.fill_between([entry[1], timestamps[timestamps.index(
                    entry[1]) + 1]], 0, 1, color=(0, 1, 0, 0.5))
            else:
                plt.fill_between([entry[1], entry[2]], 0,
                                 1, color=(0, 1, 0, 0.5))
        elif entry[0] in [51, 54]:
            if entry[1] == entry[2]:
                plt.fill_between([entry[1], timestamps[timestamps.index(
                    entry[1]) + 1]], 0, 1, color=(0, 0, 1, 0.5))
            else:
                plt.fill_between([entry[1], entry[2]], 0,
                                 1, color=(0, 0, 1, 0.5))

    ax = subplot(5)
    ticks = np.linspace(min(timestamps),
                        max(timestamps), num=10, endpoint=True)
    date_ticks = [str(datetime.fromtimestamp(round(x / 60) * 60))
                  for x in ticks]
    plt.xticks([round(x / 60) * 60 for x in ticks])
    ax.set_xticklabels(date_ticks)
    ax.set_ylim((-0.1, 1.1))
    plt.yticks([])
    plt.ylabel(ylabels[5])

    plt.annotate('Device Auth', (min(timestamps) + 15000, 0.8))
    plt.annotate('Device Theft', (min(timestamps) + 15000, 0.5))
    plt.annotate('Device Loss', (min(timestamps) + 15000, 0.2))

    activated = [-1, -1, -1, -1]
    for i in range(len(timestamps) - 1):
        random.seed(min(timestamps))
        old_state = (
            alpha[i], privacy[i], unfamiliarity[i], proximity[i])
        new_state = (
            alpha[i + 1], privacy[i + 1], unfamiliarity[i + 1], proximity[i + 1])

        result = enumerate_actions(old_state, new_state)

        if len(result) > 0:
            for res in result:
                if type(res[1]) is not str:
                    if res[1] and activated[res[0]] < 0:
                        activated[res[0]] = timestamps[i]
                    elif not res[1] and activated[res[0]] >= 0:
                        top = 1 - res[0] / 3
                        bottom = 1 - (res[0] + 1) / 3 + .05

                        plt.fill_between(
                            [activated[res[0]], timestamps[i - 1]], top, bottom, color=module_colors[res[0]], alpha=0.15)

                        activated[res[0]] = -1

    for i, x in enumerate(activated):
        if x >= 0:
            plt.fill_between(
                [activated[i], timestamps[len(timestamps) - 1]], i / 3, (i + 1) / 3, color=module_colors[i], alpha=0.15)

    plt.tight_layout()
    plt.savefig(
        'plots/student={}.pdf'.format(student))
    plt.clf()
    print(f'    Done student {student}')


if __name__ == '__main__':
    build_haggle_graphs()
