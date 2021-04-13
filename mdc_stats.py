import numpy as np
import os
from os import path
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle
from functools import reduce
from multiprocessing import Pool

from constants import *


class FormatError(Exception):
    pass


def add_alpha_values(user_file, data):
    if path.exists(path.join(ALPHA_PATH, user_file + '.pkl')):
        alpha_data = pickle.load(
            open(path.join(ALPHA_PATH, user_file + '.pkl'), 'rb'))
        high_alphas = set()
        med_alphas = set()
        for value in alpha_data.values():
            high_alpha = set(
                [x for x, y in value.items() if y > 175])
            high_alphas = high_alphas | high_alpha
            med_alpha = set(
                [x for x, y in value.items() if y > 100 and y <= 175])
            med_alphas = med_alphas | med_alpha

        data['high alpha'] = len(high_alphas)
        data['medium alpha'] = len(med_alphas)
    else:
        data['high alpha'] = 0


def calculate_overlap(user_file, data):
    visit_data_path = path.join(VISITS_PATH, user_file + '.pkl')
    scenario_data_path = path.join(PLOT_DATA_PATH, f'scenario_{user_file}.pkl')
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, f'unfamiliarity_{user_file}.pkl')
    aggregate_familiarity_path = path.join(
        PLOT_DATA_PATH, f'aggregatefam_{user_file}.pkl')

    if not path.exists(visit_data_path):
        return

    raw_visit_data = pickle.load(
        open(visit_data_path, 'rb'))
    visit_data = [x for x in raw_visit_data if x[2] == 'safe']

    _, _, familiarity_over_time, _ = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )

    aggregate_familiarity_over_time = pickle.load(
        open(aggregate_familiarity_path, 'rb'))

    scenario_data = pickle.load(
        open(scenario_data_path, 'rb')
    )

    total_safe_visit_time = sum([visit[1] - visit[0]
                                 for visit in visit_data if visit[2] == 'safe'])
    data['total safe visit time'] = total_safe_visit_time

    total_overlap = 0
    activated = 0

    for visit1 in visit_data:
        for visit2 in visit_data:
            if visit1[0] == visit2[0] and visit1[1] == visit2[1]:
                continue

            assert not (visit1[0] < visit2[1] and visit2[0]
                        < visit1[1]), f'no overlap allowed: {visit1} {visit2}'

    def calc_overlap(module_start, module_end):
        total = 0
        for visit in visit_data:
            assert visit[0] <= visit[1], f'{visit}'

            if visit[2] != 'safe':
                continue

            if visit[0] < module_end and module_start < visit[1]:
                start = module_start
                if start < visit[0]:
                    start = visit[0]

                end = module_end
                if visit[1] < end:
                    end = visit[1]

                assert end - start <= module_end - module_start
                assert end - start <= visit[1] - visit[0]

                total += (end - start)
        return total

    # Our system overlap
    latest = 0
    for timestamp in sorted(scenario_data.keys()):
        if scenario_data[timestamp][0] and activated == 0:
            activated = timestamp
        elif not scenario_data[timestamp][0] and activated != 0:
            assert activated < timestamp
            total_overlap += calc_overlap(activated, timestamp)
            activated = 0

    if activated != 0:
        assert activated <= max(scenario_data.keys())
        total_overlap += calc_overlap(activated, max(scenario_data.keys()))
        activated = 0

    data['auth overlap time'] = total_overlap
    if total_safe_visit_time:
        data['auth overlap ratio'] = total_overlap / total_safe_visit_time

    # Familiarity system overlap
    total_overlap = 0
    activated = 0
    for timestamp in familiarity_over_time.keys():
        familiarity = familiarity_over_time[timestamp]
        aggregate_familiarity = aggregate_familiarity_over_time[timestamp]
        enabled = familiarity < .85 or aggregate_familiarity < .85

        if enabled and activated == 0:
            activated = timestamp
        elif not enabled and activated != 0:
            assert activated < timestamp
            total_overlap += calc_overlap(activated, timestamp)
            activated = 0

    if activated != 0:
        end = max(familiarity_over_time.keys())
        total_overlap += calc_overlap(activated, end)
        activated = 0

    data['familiarity auth overlap time'] = total_overlap
    if total_safe_visit_time:
        data['familiarity auth overlap ratio'] = total_overlap / \
            total_safe_visit_time


def calculate_statistics(user_file, data):
    auth_data_path = path.join(AUTH_PATH, f'{user_file}.pkl')
    visit_data_path = path.join(VISITS_PATH, user_file + '.pkl')
    scenario_data_path = path.join(PLOT_DATA_PATH, f'scenario_{user_file}.pkl')
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, f'unfamiliarity_{user_file}.pkl')
    aggregate_familiarity_path = path.join(
        PLOT_DATA_PATH, f'aggregatefam_{user_file}.pkl')

    if not path.exists(visit_data_path):
        return

    raw_visit_data = pickle.load(
        open(visit_data_path, 'rb'))
    visit_data = [x for x in raw_visit_data if x[2] == 'safe']

    _, _, familiarity_over_time, _ = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )

    aggregate_familiarity_over_time = pickle.load(
        open(aggregate_familiarity_path, 'rb'))

    scenario_over_time = pickle.load(
        open(scenario_data_path, 'rb')
    )

    auth_data = pickle.load(open(auth_data_path, 'rb'))

    green_auths = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    unclass_total = 0
    unclass_positives = 0
    unclass_negatives = 0

    y_true = []
    y_pred = []

    for auth_iteration in auth_data.keys():
        safe = False
        if min(familiarity_over_time.keys()) < auth_iteration and auth_iteration < max(familiarity_over_time.keys()):
            closest = min(familiarity_over_time.keys(),
                          key=lambda x: abs(auth_iteration - x))
            familiarity = familiarity_over_time[closest]
            aggregate_familiarity = aggregate_familiarity_over_time[closest]

            enabled = familiarity < .85 or aggregate_familiarity < .85

            if not enabled:
                green_auths += 1
                safe = True

        hit = False
        if visit_data is not None:
            for visit in visit_data:
                if visit[2] == 'unknown':
                    continue

                if auth_iteration < visit[0] or visit[1] < auth_iteration:
                    continue

                if visit[2] == 'safe':
                    y_true.append(1)
                else:
                    y_true.append(0)

                if safe:
                    y_pred.append(1)
                else:
                    y_pred.append(0)

                if visit[2] == 'safe' and safe:
                    true_positives += 1
                elif visit[2] != 'safe' and safe:
                    false_positives += 1
                elif visit[2] != 'safe' and not safe:
                    true_negatives += 1
                elif visit[2] == 'safe' and not safe:
                    false_negatives += 1

                hit = True
                break

        if not hit:
            unclass_total += 1
            if safe:
                unclass_positives += 1
            else:
                unclass_negatives += 1

    start = [-1, -1, -1]
    module_total = [0, 0, 0]
    for key, value in scenario_over_time.items():
        for index, module in enumerate(value):
            if module and start[index] < 0:
                start[index] = key
            elif not module and 0 < start[index]:
                module_total[index] += key - start[index]
                start[index] = -1

    data['familiarity total class auths'] = len(auth_data) - unclass_total
    data['familiarity total unclass auths'] = unclass_total
    data['familiarity unclass ratio'] = unclass_total / len(auth_data)
    data['familiarity green auths'] = green_auths
    data['familiarity true positive auths'] = true_positives
    data['familiarity false positive auths'] = false_positives
    data['familiarity true negative auths'] = true_negatives
    data['familiarity false negative auths'] = false_negatives

    if visit_data is not None:
        all_c_pos = false_positives + true_positives
        all_c_neg = false_negatives + true_negatives
        all_g_pos = true_positives + false_negatives
        all_g_neg = true_negatives + false_positives

        if all_c_pos > 0:
            safe_precision = true_positives / all_c_pos
            data['familiarity safe auth precision'] = safe_precision

        if all_g_pos > 0:
            safe_recall = true_positives / all_g_pos
            data['familiarity safe auth recall'] = safe_recall

        if all_g_neg > 0:
            safe_fallout = false_positives / all_g_neg
            data['familiarity safe auth fpr or fallout'] = safe_fallout

        if unclass_total > 0:
            safe_unclass_fallout = unclass_positives / unclass_total
            data['familiarity safe unclassified fallout'] = safe_unclass_fallout

        if all_g_neg > 0:
            safe_tnr = true_negatives / all_g_neg
            data['familiarity safe auth tnr'] = safe_tnr

        if all_c_neg > 0:
            unsafe_precision = true_negatives / all_c_neg
            data['familiarity unsafe auth precision'] = unsafe_precision

        if all_g_neg > 0:
            unsafe_recall = true_negatives / all_g_neg
            data['familiarity unsafe auth recall'] = unsafe_recall

        if all_g_pos > 0:
            unsafe_fallout = false_negatives / all_g_pos
            data['familiarity unsafe auth fpr or fallout'] = unsafe_fallout

        if unclass_total > 0:
            unsafe_unclass_fallout = unclass_negatives / unclass_total
            data['familiarity unsafe unclassified fallout'] = unsafe_unclass_fallout

        if all_g_pos > 0:
            unsafe_tnr = true_positives / all_g_pos
            data['familiarity unsafe auth tnr'] = unsafe_tnr


def get_stats_for_file(user_file, verbose=True):
    cached_stats_path = f'stats/cached_{user_file}.pkl'
    if path.exists(cached_stats_path):
        return pickle.load(open(cached_stats_path, 'rb'))

    print(f'Handling {user_file}')
    try:
        with open(f'stats/{user_file}', 'r') as f:
            data = dict()
            lines = f.read().split('\n')
            for line in lines:
                if line == '':
                    continue

                pair = line.split('\t')
                if len(pair) != 2:
                    print(len(pair))
                    raise FormatError()

                data[pair[0]] = float(pair[1])

            print(f'  Calculating alpha for {user_file}')
            add_alpha_values(user_file, data)
            print(f'  Calculating overlap for {user_file}')
            calculate_overlap(user_file, data)
            print(f'  Calculating stats for {user_file}')
            calculate_statistics(user_file, data)
            print(f'  Done {user_file}')
            pickle.dump(data, open(cached_stats_path, 'wb'))
            return data
    except FormatError:
        print(f'  Format error in {user_file}. Skipping')
        return


def run():
    if not path.exists('stats'):
        print('No stats - aborting')
        return

    user_files = [f for f in os.listdir('stats') if not f.endswith('pkl')]
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        results = p.starmap(get_stats_for_file, [
            (user_file, False) for user_file in user_files])

    with open('stats.cfg', 'r') as config_file:
        order = [row.strip() for row in config_file.read().strip().split('\n')]

    print(f'{"key":32}{"mean":32}{"std":32}{"median":32}')
    for key in sorted(reduce(set.union, [set(res.keys()) for res in results], set()), key=lambda x: order.index(x)):
        mean = np.mean([x[key] for x in results if key in x])
        std = np.std([x[key] for x in results if key in x])
        median = np.median([x[key] for x in results if key in x])
        if 'time' in key:
            def tostr(time):
                return str(timedelta(seconds=time))
            print(f'{key:32}{tostr(mean):32}{tostr(std):32}{tostr(median):32}')
        else:
            print(f'{key:<32}{mean:<32.3f}{std:<32.3f}{median:<32.3f}')

    print('Plotting plots')

    plt.gcf().set_size_inches(14, 4)
    plt.axis('square')
    ax = plt.subplot(1, 3, 1)
    hour = 60 * 60
    ax.set_ylabel('Auth Time Enabled (Hours)')
    ax.set_xlabel('Total Usage Time (Hours)')
    ax.scatter([x['total time'] / hour for x in results],
               [x['auth time'] / hour for x in results], color='cyan')
    ax.set_ylim(ax.get_xlim())
    ax = plt.subplot(1, 3, 2)
    ax.set_ylabel('Theft Time Enabled (Hours)')
    ax.set_xlabel('Total Proximity Time (Hours)')
    ax.scatter([x['total prox time'] / hour for x in results],
               [x['theft time'] / hour for x in results], color='darkblue')
    ax.set_ylim(ax.get_xlim())
    ax = plt.subplot(1, 3, 3)
    ax.set_ylabel('Loss Time Enabled (Hours)')
    ax.set_xlabel('Total Proximity Time (Hours)')
    ax.scatter([x['total prox time'] / hour for x in results],
               [x['loss time'] / hour for x in results], color='lime')
    ax.set_ylim(ax.get_xlim())
    plt.tight_layout()
    plt.savefig('plots/time_plots.pdf')
    plt.clf()

    plt.gcf().set_size_inches(5, 5)
    data = [x['total time'] / hour for x in results]
    counts, bins = np.histogram(data, bins=18, range=(0, max(data)))
    plt.xlabel('Total Usage Time (Hours)')
    plt.hist(bins[:-1], bins, weights=counts)
    plt.tight_layout()
    plt.savefig('plots/total_time_hist.pdf')
    plt.clf()

    plt.gcf().set_size_inches(6, 3)
    x_axis = [x['total time'] /
              hour for x in sorted(results, key=lambda z: z['total time'])]
    y_axis = [x['success ratio'] * 100
              for x in sorted(results, key=lambda z: z['total time'])]
    y_line = np.poly1d(np.polyfit(x_axis, y_axis, 1))(np.unique(x_axis))

    print(f'Best fit endpoints\t\t{min(y_line):.2%} -> {max(y_line):.2%}')

    plt.scatter(x_axis, y_axis, color='blue')
    plt.plot(np.unique(x_axis), y_line, color='red')
    plt.ylim(0, 100)
    plt.xlabel('Total Usage Time (Hours)')
    plt.ylabel('Success Ratio (%)')
    plt.tight_layout()
    plt.savefig('plots/success_ratio.pdf')
    plt.clf()

    y_axes = list()
    for user_file in user_files:
        auth_data = pickle.load(
            open(path.join(AUTH_PATH, f'computed_{user_file}.pkl'), 'rb'))
        x_axis = sorted(list(auth_data.keys()))
        y_axis = list()
        current_success = 0
        current_total = 0
        for timestamp in x_axis:
            if auth_data[timestamp]:
                current_success += 1
                current_total += 1
                y_axis.append(current_success / current_total)
            else:
                current_total += 1
                y_axis.append(current_success / current_total)

        y_axes.append(y_axis)

    max_len = max([len(ax) for ax in y_axes])
    means = list()
    for index in range(max_len):
        values = list()
        for axis in y_axes:
            if len(axis) > index:
                values.append(axis[index])

        means.append((sum(values) / len(values)) * 100)

    plt.plot([x for x in range(len(means))][50:], means[50:])
    plt.ylim(0, 100)
    plt.xlabel('Total Number of Authentications')
    plt.ylabel('Success Ratio (%)')
    plt.tight_layout()
    plt.savefig('plots/success_over_auth.pdf')
    plt.clf()


if __name__ == '__main__':
    run()
