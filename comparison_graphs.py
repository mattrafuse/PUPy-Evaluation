import os
from os import path
import random
import pickle
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from datetime import datetime

from actions import enumerate_actions
from constants import BLACKLIST
from util_fns import Ticker, get_file_size
from constants import *

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
    'Alpha',
    'Privacy',
    'Unfamiliarity',
    'Proximity',
    'Actions'
]


def generate_context_for_user(user_file, interval_size, verbose):
    context_data_path = path.join(
        PLOT_DATA_PATH, 'context_' + user_file)

    if path.exists(context_data_path):
        print(f'  Generating context from file {user_file} (Skipped)')
        return
    else:
        print(f'  Generating context from file {user_file}')

    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    if not path.exists(unfamiliarity_data_path):
        print('    ERROR: You need to build mdc graphs first.')
        return
    else:
        familiarity_over_time = pickle.load(
            open(unfamiliarity_data_path, 'rb'))[2]

    gps_user_data = pickle.load(
        open(path.join(GPS_USER_PATH, user_file), 'rb'))
    gpswlan_user_data = pickle.load(
        open(path.join(GPSWLAN_USER_PATH, user_file), 'rb'))

    ticker = Ticker(len(familiarity_over_time))

    context_over_time = dict()
    for timestamp in familiarity_over_time.keys():
        lower_bound = timestamp
        upper_bound = timestamp + interval_size

        found_contexts = dict()
        for record in gps_user_data[:]:
            if lower_bound <= record[4] and record[4] < upper_bound:
                gps_user_data.remove(record)
                coord = (int(record[0] * 1000), int(record[1] * 1000))
                context_over_time[timestamp] = str(coord)
                break

        if context_over_time.get(timestamp, None) is not None:
            continue

        for record in gpswlan_user_data[:]:
            if lower_bound <= record[3] and record[3] < upper_bound:
                gpswlan_user_data.remove(record)
                coord = (int(record[0] * 1000), int(record[1] * 1000))
                context_over_time[timestamp] = str(coord)
                break

        if verbose:
            remaining = len(gps_user_data) + len(gpswlan_user_data)
            ticker.tick('      ', f'{remaining} entries remaining')
        else:
            ticker.tick('      ')

    pickle.dump(context_over_time, open(context_data_path, 'wb'))
    print(f'  Done calculating context for {user_file}.')


def occ(timestamp, context_over_time, familiarity_over_time):
    learning_rate = 0.05
    if context_over_time.get(timestamp, None) is None:
        return 0

    current_context = context_over_time[timestamp]

    timestamps = sorted(context_over_time.keys(),
                        key=lambda x: int(x))

    result = 0
    while len(timestamps) > 0:
        if int(timestamps[0]) > int(timestamp):
            break

        if context_over_time[timestamps[0]] == current_context:
            result = learning_rate * \
                familiarity_over_time[timestamps[0]] + \
                (1 - learning_rate) * result
        timestamps = timestamps[1:]

    return result


def generate_graph_for_user(user_file, verbose=True):
    if not path.exists(PLOT_DATA_PATH):
        os.mkdir(PLOT_DATA_PATH)

    print(f'  Generating comparison graph from file {user_file}')
    print(f'    {user_file} - Loading files                                      ')
    alpha_data_path = path.join(PLOT_DATA_PATH, 'alpha_' + user_file)
    if not path.exists(alpha_data_path):
        print('    ERROR: You need to build mdc graphs first.')
        return
    else:
        alpha_over_time = pickle.load(open(alpha_data_path, 'rb'))

    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    if not path.exists(unfamiliarity_data_path):
        print('    ERROR: You need to build mdc graphs first.')
        return
    else:
        privacy_over_time, unfamiliarity_over_time, familiarity_over_time, _ = pickle.load(
            open(unfamiliarity_data_path, 'rb')
        )

    context_data_path = path.join(PLOT_DATA_PATH, 'context_' + user_file)
    if not path.exists(unfamiliarity_data_path):
        print('    ERROR: You need to build context first.')
        return
    else:
        context_over_time = pickle.load(open(context_data_path, 'rb'))

    proximity_data_path = path.join(PLOT_DATA_PATH, 'proximity_' + user_file)
    if not path.exists(proximity_data_path):
        print('    ERROR: You need to build mdc graphs first.')
        return
    else:
        proximity_over_time = pickle.load(open(proximity_data_path, 'rb'))

    print(
        f'    {user_file} - Computing scenario results                                ')
    ticker = Ticker(len(unfamiliarity_over_time), accuracy=1)
    scenario_over_time = dict()
    aggregate_familiarity_path = path.join(
        PLOT_DATA_PATH, 'aggregatefam_' + user_file)
    if path.exists(aggregate_familiarity_path):
        aggregate_familiarity_over_time = pickle.load(
            open(aggregate_familiarity_path, 'rb'))
    else:
        aggregate_familiarity_over_time = dict()

    old_state = None
    activated = [0, 0]
    for key in unfamiliarity_over_time.keys():
        if min(alpha_over_time.keys()) < key and key < max(alpha_over_time.keys()):
            closest = min(alpha_over_time.keys(), key=lambda x: abs(key - x))
            alpha = alpha_over_time[closest]
        else:
            alpha = 3

        privacy = privacy_over_time[key]
        unfamiliarity = unfamiliarity_over_time[key]
        familiarity = familiarity_over_time[key]
        if context_over_time.get(key, None) is None:
            familiarity = 0

        if not path.exists(aggregate_familiarity_path):
            aggregate_familiarity_over_time[key] = occ(key,
                                                       context_over_time, familiarity_over_time)
        aggregate_familiarity = aggregate_familiarity_over_time[key]

        if min(proximity_over_time.keys()) < key and key < max(proximity_over_time.keys()):
            closest = min(proximity_over_time.keys(),
                          key=lambda x: abs(key - x))
            proximity = proximity_over_time[closest]
        else:
            proximity = 1.0

        new_state = (alpha, privacy, unfamiliarity, proximity)

        if old_state is not None:
            # print(new_state, enumerate_actions(old_state, new_state))
            for item in enumerate_actions(old_state, new_state):
                if item[0] == 0:
                    activated[item[0]] = item[1]
            if familiarity < .4 or aggregate_familiarity < .4:
                activated[1] = 2
            elif familiarity < .85 or aggregate_familiarity < .85:
                activated[1] = 1
            # activated[1] = familiarity < .85 or aggregate_familiarity < .85

        scenario_over_time[key] = activated[:]
        old_state = new_state
        ticker.tick('      ')

    if not path.exists(aggregate_familiarity_path):
        pickle.dump(aggregate_familiarity_over_time,
                    open(aggregate_familiarity_path, 'wb'))

    if user_file in BLACKLIST:
        plt.clf()
        print(f'    {user_file} - Blacklisted                                ')
        return

    print(f'    {user_file} - Plotting data                                ')
    plt.gcf().set_size_inches(75, 20)
    plt.gcf().set_dpi(200.0)
    alpha_ax = plt.subplot(6, 1, 1)
    alpha_ax.set_ylabel('Context Familiarity')
    privacy_ax = plt.subplot(6, 1, 2)
    privacy_ax.set_ylabel('Privacy')
    unfamiliarity_ax = plt.subplot(6, 1, 3)
    unfamiliarity_ax.set_ylabel('Unfamiliarity')
    proximity_ax = plt.subplot(6, 1, 4)
    proximity_ax.set_ylabel('Proximity')
    familiarity_ax = plt.subplot(6, 1, 5)
    familiarity_ax.set_ylabel('Familiarity')
    actions_ax = plt.subplot(6, 1, 6)
    actions_ax.set_ylabel('Modules')
    actions_ax.set_xlabel('Time (Days)')

    min_timestamp = max(
        min(alpha_over_time.keys()),
        min(unfamiliarity_over_time.keys()),
        min(proximity_over_time.keys())
    )

    max_timestamp = min(
        max(alpha_over_time.keys()),
        max(unfamiliarity_over_time.keys()),
        max(proximity_over_time.keys())
    )

    for ax in [alpha_ax, privacy_ax, unfamiliarity_ax, proximity_ax, familiarity_ax, actions_ax]:
        ax.set_xlim(min_timestamp - DAY / 2, max_timestamp + DAY / 2)
        ax.set_ylim(0.0, 1.0)

        num_days = round(max_timestamp / DAY) - round(min_timestamp / DAY)

        ticks = np.linspace(round(min_timestamp / DAY) * DAY, round(max_timestamp / DAY) * DAY,
                            num=num_days, endpoint=True)
        # date_ticks = [str(datetime.fromtimestamp(round(x / DAY) * DAY))
        #               for x in ticks]
        date_ticks = [str(x + 1) for x in range(len(ticks))]
        ax.set_xticks(ticks)
        ax.set_xticklabels(date_ticks)
        ax.grid(axis='x')
    alpha_ax.set_ylim(0, 205)

    print(f'    {user_file} - Plotting values                                ')
    alpha_ax.plot(list(alpha_over_time.keys()), list(
        alpha_over_time.values()), color='purple')
    privacy_ax.plot(list(unfamiliarity_over_time.keys()),
                    list(privacy_over_time.values()), color='blue')
    unfamiliarity_ax.plot(list(privacy_over_time.keys()), list(
        unfamiliarity_over_time.values()), color='green')
    proximity_ax.plot(list(proximity_over_time.keys()), list(
        proximity_over_time.values()), color='gold')
    familiarity_ax.plot(list(familiarity_over_time.keys()), list(
        familiarity_over_time.values()), color='orange')
    familiarity_ax.plot(list(aggregate_familiarity_over_time.keys()), list(
        aggregate_familiarity_over_time.values()), color='red')

    print(f'    {user_file} - Plotting actions                                ')
    start = [-1, -1]
    current_state = [0, 0]
    for key, value in scenario_over_time.items():
        for index, module in enumerate(value):
            ylims = (
                1 - (index / len(value)),
                (1 - (index + 1) / len(value))
            )
            alpha = current_state[index] * .33

            if module and start[index] < 0:
                start[index] = key
                current_state[index] = module

            elif module and module != current_state[index]:
                actions_ax.fill_between(
                    [start[index], key], ylims[0], ylims[1], color='cyan', alpha=alpha)
                start[index] = key
                current_state[index] = module

            elif not module and 0 < start[index]:
                actions_ax.fill_between(
                    [start[index], key], ylims[0], ylims[1], color='cyan', alpha=alpha)
                start[index] = -1
                current_state[index] = -1

    for index, val in enumerate(start):
        if val > -1:
            actions_ax.fill_between([val, max(scenario_over_time.keys())], 1 - (index / len(
                start)), (1 - (index + 1) / len(start)), color='cyan', alpha=0.25)

    if not path.exists('plots'):
        os.mkdir('plots')

    plt.tight_layout()
    plt.savefig(path.join('plots', 'comparison_' +
                          path.splitext(user_file)[0] + '.pdf'))
    plt.clf()
    print(f'    {user_file} - Done                                ')


def generate_stats_for_user(user_file, verbose=True):
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    aggregate_familiarity_path = path.join(
        PLOT_DATA_PATH, 'aggregatefam_' + user_file)

    _, _, familiarity_over_time, _ = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )

    aggregate_familiarity_over_time = pickle.load(
        open(aggregate_familiarity_path, 'rb')
    )

    familiarity_auth_total = 0
    activated = 0
    for timestamp in familiarity_over_time.keys():
        familiarity = familiarity_over_time[timestamp]
        aggregate_familiarity = aggregate_familiarity_over_time[timestamp]
        if activated == 0 and (familiarity < .85 or aggregate_familiarity < .85):
            activated = timestamp
        elif activated and not (familiarity < .85 or aggregate_familiarity < .85):
            familiarity_auth_total += (timestamp - activated)
            activated = 0

    if activated != 0:
        familiarity_auth_total += (max(familiarity_over_time.keys()) - activated)
        activated = 0

    total_time = max(familiarity_over_time.keys()) - \
        min(familiarity_over_time.keys())

    stats_path = f'stats/{path.splitext(user_file)[0]}'
    if path.exists(stats_path):
        f = open(stats_path, 'a')
    else:
        f = open(stats_path, 'w')
    f.write(f'familiarity auth time\t{familiarity_auth_total}\n')
    f.write(
        f'familiarity auth ratio\t{familiarity_auth_total / total_time}\n')
    f.close()


def build_comparison_graphs():
    folders = [os.listdir(x) for x in [BT_USER_PATH,
                                       GPS_USER_PATH, GPSWLAN_USER_PATH, ACCEL_USER_PATH, APP_USER_PATH]]
    files_intersect = set(folders[0])
    for folder in folders:
        files_intersect = files_intersect & set(folder)
    user_files = sorted(list(files_intersect), key=get_file_size)[9:]
    interval = 30 * 60  # 30 minute intervals

    print('Generating context')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_context_for_user, [
                  (user_file, interval, False) for user_file in user_files])

    print('Generating graphs')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_graph_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Generating stats')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_stats_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Script Complete.')


if __name__ == '__main__':
    build_comparison_graphs()
