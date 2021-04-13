import os
from os import path
import random
import pickle
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
from datetime import datetime
from constants import *
from sklearn.metrics import precision_score, recall_score

from actions import enumerate_actions
from util_fns import Ticker, get_file_size

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


def parse_files(filenames, sizes):
    parent = parse_file(filenames[0], sizes[0])
    child = parse_file(filenames[1], sizes[1])

    for key in parent.keys():
        if filenames[0].startswith('bt'):
            parent[key] = [child[parent[key][0]][0]]
        # This seems to be the preferred orientation for the cases it's used
        else:
            parent[key] = child[parent[key][0]][1:] + parent[key][1:]

    return parent


def parse_file(filename, size, index=0):
    result = dict()
    print(f'\n  Processing {filename} data')
    fullpath = path.join('datasets/mdc/mdcdb', filename)

    with open(fullpath) as f:
        ticker = Ticker(size)
        while True:
            line = f.readline().strip()
            if line == '':
                break
            data = line.split('\t')

            result[data[index]] = list(data[index+1:])

            ticker.tick('    ')
    return result


def build_pkl_files():
    records_file = 'datasets/mdc/mdcdb/records.csv'

    gps = parse_file('gps.csv', 11077061)
    wlan = parse_files(['wlanrelation.csv', 'wnetworks.csv'], [
                       53432599, 590994])
    gpswlan = parse_file('gpswlan.csv', 11302232)
    bt = parse_files(['btrelation.csv', 'bnetworks.csv'], [31801059, 586688])
    app = parse_file('application.csv', 8292063)

    NUM_RECORDS = 103799831
    print("\n  Processing records.csv data")
    with open(records_file) as f:
        ticker = Ticker(NUM_RECORDS)
        while True:
            line = f.readline().strip()
            if line == '':
                break

            data = line.split('\t')

            if data[4] == 'gps' and data[0] in gps:
                gps[data[0]] = gps[data[0]] + list(data[1:4])
            elif data[4] == 'wlan' and data[0] in wlan:
                wlan[data[0]] = wlan[data[0]] + list(data[1:4])
            elif data[4] == 'gpswlan' and data[0] in gpswlan:
                gpswlan[data[0]] = gpswlan[data[0]] + list(data[1:4])
            elif data[4] == 'bt' and data[0] in bt:
                bt[data[0]] = bt[data[0]] + list(data[1:4])
            elif data[4] == 'app' and data[0] in app:
                app[data[0]] = app[data[0]] + list(data[1:4])

            ticker.tick('    ')

    for result in [gps, wlan, gpswlan, bt, app]:
        filename = [k for k, v in locals().items() if v == result][0]
        print(f'Writing {filename}.pkl')
        with open(f'datasets/mdc/parsed/{filename}.pkl', 'wb') as f:
            pickle.dump(result, f)


def build_visits_pkls():
    print('Splitting visit data')
    places_file = 'datasets/mdc/mdcdb/places.csv'
    user_places = dict()
    with open(places_file) as places:
        while True:
            line = places.readline().strip()
            if line == '':
                break

            place = line.split('\t')

            if user_places.get(place[0], None) is None:
                user_places[place[0]] = dict()

            safe = ['1', '3']

            unsafe = ['4', '6', '7', '8', '8', '10']

            if place[2] in safe:
                user_places[place[0]][place[1]] = 'safe'
            elif place[2] in unsafe:
                user_places[place[0]][place[1]] = 'unsafe'
            else:
                user_places[place[0]][place[1]] = 'unknown'

    visits10_file = 'datasets/mdc/mdcdb/visits_10min.csv'
    visits20_file = 'datasets/mdc/mdcdb/visits_20min.csv'

    users = dict()

    def parse_visits(visits):
        while True:
            line = visits.readline().strip()
            if line == '':
                break

            visit = line.split('\t')

            if user_places.get(visit[0], None) is None:
                continue
            elif user_places[visit[0]].get(visit[1], None) is None:
                continue

            if users.get(visit[0], None) is None:
                users[visit[0]] = list()

            userid = visit[0]
            placeid = visit[1]
            start_timestamp = int(visit[2]) + int(visit[3])
            end_timestamp = int(visit[4]) + int(visit[5])

            if end_timestamp < start_timestamp:
                print('  Discrepancy found!')
                tmp = start_timestamp
                start_timestamp = end_timestamp
                end_timestamp = tmp

            users[userid].append(
                (start_timestamp, end_timestamp, user_places[userid][placeid]))

    with open(visits10_file) as visits10:
        parse_visits(visits10)

    with open(visits20_file) as visits20:
        parse_visits(visits20)

    if not path.exists(VISITS_PATH):
        os.mkdir(VISITS_PATH)

    for userid in users.keys():
        raw_visit_data = users[userid]

        visit_data = []
        current_max = 0
        for visit in sorted(raw_visit_data, key=lambda x: x[0]):
            if visit[1] <= current_max:
                continue

            visit_start = visit[0]
            if visit[0] < current_max:
                visit_start = current_max

            visit_data.append((visit_start, visit[1], visit[2]))
            current_max = visit[1]

        with open(f'{VISITS_PATH}/{userid}.pkl', 'wb') as f:
            pickle.dump(visit_data, f)

    print('  Done                              ')


def build_accel_pkl():
    records_2_file = 'datasets/mdc/mdcdb/records_2.csv'
    accel = parse_file('accel_activity_noko.csv', 1063725, 1)

    NUM_RECORDS = 1215615
    print("\n  Processing records_2.csv data")
    with open(records_2_file) as f:
        ticker = Ticker(NUM_RECORDS)
        while True:
            line = f.readline().strip()
            if line == '':
                break

            data = line.split('\t')
            if data[0] in accel:
                accel[data[0]] = accel[data[0]] + list(data[1:4])
            ticker.tick('    ')

    with open(f'datasets/mdc/parsed/accel.pkl', 'wb') as f:
        pickle.dump(accel, f)

# def build_places_pkl():
#     places_file, places_length = 'datasets/mdc/mdcdb/places.csv', 493
#     visits_10min_file = 'datasets/mdc/mdcdb/visits_10min.csv'
#     visits_20min_file = 'datasets/mdc/mdcdb/visits_20min.csv'

#     places = dict()
#     with open(places_file) as f:


def split_data(name):
    print(f'Splitting {name} data')
    parsed_path = f'{PARSED_PATH}/{name}.pkl'
    user_path = f'{PARSED_PATH}/{name}_by_user'

    print(f'  Loading {parsed_path}')
    data = pickle.load(open(parsed_path, 'rb'))

    print(f'  Parsing {name} data')
    ticker = Ticker(len(data))
    users = dict()

    for key, value in data.items():
        ticker.tick('    ')
        if len(value) > 1:
            if name == 'bt':
                users[value[1]] = users.get(
                    value[1], list()) + [(value[0], int(value[2]) + int(value[3]))]
            elif name == 'gps':
                new_entry = (
                    float(value[1]),  # Lat
                    float(value[2]),  # Lng
                    float(value[3] if value[3] != '\\N' else 0),  # Altitude
                    float(value[4] if value[4] != '\\N' else 0),  # Speed
                    int(value[12]) + int(value[13])  # Timestamp
                )
                users[value[11]] = users.get(value[11], list()) + [new_entry]
            elif name == 'accel':
                if len(value) > 3 and value[2] == 't':
                    users[value[3]] = users.get(
                        value[3], list()) + [(value[0], int(value[4]) + int(value[5]))]
            elif name == 'app':
                if len(value) > 4:
                    new_entry = (
                        value[0],  # event
                        value[2],  # app name
                        int(value[5]) + int(value[6])  # timestamp
                    )
                    users[value[4]] = users.get(value[4], list()) + [new_entry]
            elif name == 'wlan':
                users[value[5]] = users.get(
                    value[5], list()) + [(value[0], int(value[6]) + int(value[7]))]
            elif name == 'gpswlan':
                if len(value) > 4:
                    new_entry = (
                        float(value[0]),  # Lat
                        float(value[1]),  # Lng
                        value[2],  # mac
                        int(value[4]) + int(value[5])  # Timestamp
                    )
                    users[value[3]] = users.get(value[3], list()) + [new_entry]

    print('\n  Writing individual user files')
    if not path.exists(user_path):
        os.mkdir(user_path)

    ticker = Ticker(len(users))
    for key, value in users.items():
        ticker.tick('    ')
        pickle.dump(value, open(f'{user_path}/{key}.pkl', 'wb'))

    print('\n  Done')


def generate_auth_for_user(user_file, verbose=True):
    if path.exists(path.join(AUTH_PATH, user_file)):
        print(f'  Generating auth data from file {user_file} (SKIPPED)')
        return

    print(f'  Generating auth data from file {user_file}')
    if verbose:
        print('    Loading file')
    user_data = pickle.load(open(path.join(APP_USER_PATH, user_file), 'rb'))

    user_data = sorted(user_data, key=lambda x: x[1])

    if verbose:
        print('    Calculating auth points')
    ticker = Ticker(len(user_data))
    iteration_results = dict()
    previous = None
    for event in user_data:
        if previous is None:
            iteration_results[event[2]] = True
        # From https://www.usenix.org/system/files/conference/soups2014/soups14-paper-harbach.pdf
        # One standard deviation from the mean
        elif (event[2] - previous[2]) > 70 + 1 * 240:
            iteration_results[event[2]] = True
        previous = event
        if verbose:
            ticker.tick('      ')

    if verbose:
        print('\n    Writing results')
    if not path.exists(AUTH_PATH):
        os.mkdir(AUTH_PATH)

    if verbose:
        print('    interation_results length:', len(iteration_results))
    pickle.dump(iteration_results, open(path.join(AUTH_PATH, user_file), 'wb'))
    print(f'    Done generating auth data from file {user_file}')


def occ(address, previous_iterations):
    learning_rate = 0.05
    last_seen = 0
    timestamps = sorted(previous_iterations.keys(),
                        key=lambda x: int(x), reverse=True)
    hits = list()
    while len(timestamps) > 0:
        if last_seen > 2000:
            break

        if previous_iterations[timestamps[0]].get(address, -1) == -1:
            last_seen += 1
            hits.append(0)
        else:
            hits.append(1)
            last_seen = 0

        timestamps = timestamps[1:]

    total = 0
    for i in range(1, len(hits)):
        total = learning_rate * hits[-i] + (1 - learning_rate) * total

    return learning_rate + (1 - learning_rate) * total


def generate_unfamiliarity_for_user(user_file, interval_size, verbose=True):
    if path.exists(path.join(UNFAMILIARITY_PATH, user_file)):
        print(
            f'  Generating unfamiliarity data from file {user_file} (SKIPPED)')
        return

    print(f'  Generating unfamiliarity data from file {user_file}')
    if verbose:
        print('    Loading file')
    user_data = pickle.load(open(path.join(BT_USER_PATH, user_file), 'rb'))

    if verbose:
        print('    Calculating timestamps and intervals')
    earliest_timestamp = min([x[1] for x in user_data])
    latest_timestamp = max(x[1] for x in user_data)
    if verbose:
        print('      Earliest timestamp:', earliest_timestamp)
    if verbose:
        print('      Latest timestamp:', latest_timestamp)

    n_intervals = math.ceil(
        (latest_timestamp - earliest_timestamp) / interval_size) - 1
    if verbose:
        print(
            f'      Based on interval of size {interval_size}s, there are {n_intervals} intervals')

    if verbose:
        print('    Calculating unfamiliarity')
    if verbose:
        ticker = Ticker(n_intervals)
    else:
        ticker = Ticker(n_intervals, accuracy=1)
    iteration_results = dict()
    for iteration in range(n_intervals):
        lower_bound = earliest_timestamp + iteration * interval_size
        upper_bound = earliest_timestamp + (iteration + 1) * interval_size

        found_addresses = dict()
        for record in user_data[:]:
            if lower_bound <= record[1] and record[1] < upper_bound:
                user_data.remove(record)
                found_addresses[record[0]] = occ(
                    record[0], iteration_results.copy())

        iteration_results[lower_bound] = found_addresses
        if verbose:
            ticker.tick('      ', f'{len(user_data)} entries remaining')
        else:
            ticker.tick('      ')

    if verbose:
        print('\n    Writing results')
    if not path.exists(UNFAMILIARITY_PATH):
        os.mkdir(UNFAMILIARITY_PATH)

    if verbose:
        print('    interation_results length:', len(iteration_results))
    pickle.dump(iteration_results, open(
        path.join(UNFAMILIARITY_PATH, user_file), 'wb'))
    print(f'    Done generating unfamiliarity data from file {user_file}')


def generate_proximity_for_user(user_file, interval_size, verbose=True):
    if path.exists(path.join(PROXIMITY_PATH, user_file)):
        print(f'  Generating proximity data from file {user_file} (SKIPPED)')
        return

    print(f'  Generating proximity data from file {user_file}')
    if verbose:
        print('    Loading file')
    user_data = pickle.load(open(path.join(ACCEL_USER_PATH, user_file), 'rb'))

    if verbose:
        print('    Calculating timestamps and intervals')
    earliest_timestamp = min([x[1] for x in user_data])
    latest_timestamp = max(x[1] for x in user_data)
    if verbose:
        print('      Earliest timestamp:', earliest_timestamp)
    if verbose:
        print('      Latest timestamp:', latest_timestamp)

    n_intervals = math.ceil(
        (latest_timestamp - earliest_timestamp) / interval_size) - 1
    if verbose:
        print(
            f'      Based on interval of size {interval_size}s, there are {n_intervals} intervals')

    if verbose:
        print('    Calculating proximity')
    ticker = Ticker(n_intervals)
    iteration_results = dict()
    for iteration in range(n_intervals):
        lower_bound = earliest_timestamp + iteration * interval_size
        upper_bound = earliest_timestamp + (iteration + 1) * interval_size

        found_proximity = list()
        for record in user_data[:]:
            if lower_bound <= record[1] and record[1] < upper_bound:
                user_data.remove(record)
                found_proximity.append(record[0])

        # Kind of a hack to get proximity data across the board
        if len(found_proximity) == 0:
            key_index = random.randint(0, len(iteration_results) - 1)
            copied_key = list(iteration_results.keys())[key_index]
            iteration_results[lower_bound] = iteration_results[copied_key].copy()
        else:
            iteration_results[lower_bound] = found_proximity

        if verbose:
            ticker.tick('      ', f'{len(user_data)} entries remaining')

    if verbose:
        print('\n    Writing results')
    if not path.exists(PROXIMITY_PATH):
        os.mkdir(PROXIMITY_PATH)

    if verbose:
        print('    interation_results length:', len(iteration_results))
    pickle.dump(iteration_results, open(
        path.join(PROXIMITY_PATH, user_file), 'wb'))
    print('    Done generating proximity data from file {user_file}')


def alpha_occ(coord, previous_iterations):
    last_seen = 0
    total = 1

    timestamps = sorted(previous_iterations.keys(),
                        key=lambda x: int(x), reverse=True)
    while len(timestamps) > 0:
        if last_seen > 2000:
            break

        if total >= 200:
            return 200

        if previous_iterations[timestamps[0]].get(coord, -1) != -1:
            total += 1
            last_seen = 0
        else:
            last_seen += 1

        timestamps = timestamps[1:]

    return total


def generate_alpha_for_user(user_file, interval_size, verbose=True):
    if path.exists(path.join(ALPHA_PATH, user_file)):
        print(f'  Generating alpha data from file {user_file} (SKIPPED)')
        return

    print(f'  Generating alpha data from file {user_file}')
    if verbose:
        print('    Loading file')
    # A Hack to remove non-timestamp data
    gps_user_data = list(filter(lambda x: len(x) >= 5, pickle.load(
        open(path.join(GPS_USER_PATH, user_file), 'rb'))))
    wlan_user_data = pickle.load(
        open(path.join(WLAN_USER_PATH, user_file), 'rb'))
    gpswlan_user_data = pickle.load(
        open(path.join(GPSWLAN_USER_PATH, user_file), 'rb'))

    if verbose:
        print('    Calculating timestamps and intervals')
    earliest_timestamp = min(
        min([x[4] for x in gps_user_data]),
        min([x[1] for x in wlan_user_data]),
        min([x[3] for x in gpswlan_user_data])
    )

    latest_timestamp = max(
        max([x[4] for x in gps_user_data]),
        max([x[1] for x in wlan_user_data]),
        max([x[3] for x in gpswlan_user_data])
    )

    if verbose:
        print('      Earliest timestamp:', earliest_timestamp)
    if verbose:
        print('      Latest timestamp:', latest_timestamp)

    n_intervals = math.ceil(
        (latest_timestamp - earliest_timestamp) / interval_size) - 1
    if verbose:
        print(
            f'      Based on interval of size {interval_size}s, there are {n_intervals} intervals')

    if verbose:
        print('    Calculating alpha')
    if verbose:
        ticker = Ticker(n_intervals)
    else:
        ticker = Ticker(n_intervals, accuracy=1)
    iteration_results = dict()
    for iteration in range(n_intervals):
        lower_bound = earliest_timestamp + iteration * interval_size
        upper_bound = earliest_timestamp + (iteration + 1) * interval_size

        found_contexts = dict()
        for record in gps_user_data[:]:
            if lower_bound <= record[4] and record[4] < upper_bound:
                gps_user_data.remove(record)
                coord = (int(record[0] * 1000), int(record[1] * 1000))
                found_contexts[coord] = alpha_occ(
                    coord, iteration_results.copy())

        for record in wlan_user_data[:]:
            if lower_bound <= record[1] and record[1] < upper_bound:
                wlan_user_data.remove(record)
                found_contexts[record[0]] = alpha_occ(
                    record[0], iteration_results.copy())

        for record in gpswlan_user_data[:]:
            if lower_bound <= record[3] and record[3] < upper_bound:
                gpswlan_user_data.remove(record)
                coord = (int(record[0] * 1000), int(record[1] * 1000))
                found_contexts[coord] = alpha_occ(
                    coord, iteration_results.copy())

        iteration_results[lower_bound] = found_contexts
        if verbose:
            remaining = len(gps_user_data) + \
                len(wlan_user_data) + len(gpswlan_user_data)
            ticker.tick('      ', f'{remaining} entries remaining')
        else:
            ticker.tick('      ')

    if verbose:
        print('\n    Writing results')
    if not path.exists(ALPHA_PATH):
        os.mkdir(ALPHA_PATH)

    if verbose:
        print('    interation_results length:', len(iteration_results))
    pickle.dump(iteration_results, open(
        path.join(ALPHA_PATH, user_file), 'wb'))
    if verbose:
        print('    Done')


def generate_plot_data_for_user(user_file, verbose=True):
    proximity_alpha = 10.0
    proximity_limit = 1.0

    print(f'  Generating plot data from file {user_file}')
    unfamiliarity_data = pickle.load(
        open(path.join(UNFAMILIARITY_PATH, user_file), 'rb'))
    alpha_data = pickle.load(open(path.join(ALPHA_PATH, user_file), 'rb'))
    proximity_data = pickle.load(
        open(path.join(PROXIMITY_PATH, user_file), 'rb'))

    privacy_over_time = dict()
    familiarity_over_time = dict()
    unfamiliarity_over_time = dict()
    alpha_over_time = dict()
    proximity_over_time = dict()
    highlight_over_time = dict()

    alpha_data_path = path.join(PLOT_DATA_PATH, 'alpha_' + user_file)
    if not path.exists(alpha_data_path):
        print(
            f'    {user_file} - Computing alpha                                    ')
        for key, alpha_iteration in alpha_data.items():
            if len(alpha_iteration) > 0:
                context_alpha = round(
                    sum(alpha_iteration.values()) / len(alpha_iteration))
            else:
                context_alpha = 3

            if context_alpha < 3:
                context_alpha = 3

            alpha_over_time[key] = context_alpha
        pickle.dump(alpha_over_time, open(alpha_data_path, 'wb'))
    else:
        alpha_over_time = pickle.load(open(alpha_data_path, 'rb'))
        print(
            f'    {user_file} - Computing alpha (cached)                           ')

    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    if not path.exists(unfamiliarity_data_path):
        print(
            f'    {user_file} - Computing unfamiliarity and privacy                ')
        ticker = Ticker(len(unfamiliarity_data), accuracy=1)
        for key, unfamiliarity_iteration in unfamiliarity_data.items():
            if min(alpha_data.keys()) < key and key < max(alpha_data.keys()):
                closest = min(alpha_data.keys(), key=lambda x: abs(key - x))
                if len(alpha_data[closest]) > 0:
                    context_alpha = round(
                        sum(alpha_data[closest].values()) / len(alpha_data[closest]))
                else:
                    context_alpha = 3
            else:
                context_alpha = 3

            if len(unfamiliarity_iteration) > 0:
                context_familiarity = (
                    1 / len(unfamiliarity_iteration)) * sum(unfamiliarity_iteration.values())
            else:
                continue

            context_unfamiliarity = len(
                unfamiliarity_iteration) * (1 - context_familiarity)
            privacy = 1 - \
                ((context_alpha ** (1 - 1 /
                                    (len(unfamiliarity_iteration) + 0.00001))) / context_alpha)
            unfamiliarity = (
                context_alpha ** (1 - 1 / (context_unfamiliarity + 0.00001))) / context_alpha

            privacy_over_time[key] = privacy
            familiarity_over_time[key] = context_familiarity
            unfamiliarity_over_time[key] = unfamiliarity

            if privacy + unfamiliarity < .75:
                highlight_over_time[key] = True
            else:
                highlight_over_time[key] = False

            ticker.tick('      ')

        payload = (privacy_over_time, unfamiliarity_over_time,
                   familiarity_over_time, highlight_over_time)
        pickle.dump(payload, open(unfamiliarity_data_path, 'wb'))
    else:
        privacy_over_time, unfamiliarity_over_time, familiarity_over_time, highlight_over_time = pickle.load(
            open(unfamiliarity_data_path, 'rb')
        )
        print(
            f'    {user_file} - Computing unfamiliarity and privacy (cached)       ')

    proximity_data_path = path.join(PLOT_DATA_PATH, 'proximity_' + user_file)
    if not path.exists(proximity_data_path):
        print(
            f'    {user_file} - Computing proximity                                ')
        for key, proximity_iteration in proximity_data.items():
            if len(proximity_iteration) > 0:
                proximity_sum = 0
                for entry in proximity_iteration:
                    if entry in ['walk', 'run']:
                        proximity_sum += 0
                    elif entry in ['bicycle']:
                        proximity_sum += 1
                    elif entry in ['train/metro/tram']:
                        proximity_sum += 3
                    else:
                        proximity_sum += 5

                mean_proximity = proximity_sum / len(proximity_iteration)

            else:
                continue

            if mean_proximity - proximity_limit <= 0:
                context_proximity = 1
            else:
                exponent = (1 - 1 / (mean_proximity -
                                     proximity_limit + 0.00001))
                context_proximity = 1 - \
                    ((proximity_alpha ** exponent) / proximity_alpha)

            proximity_over_time[key] = context_proximity
        pickle.dump(proximity_over_time, open(proximity_data_path, 'wb'))
    else:
        proximity_over_time = pickle.load(open(proximity_data_path, 'rb'))
        print(
            f'    {user_file} - Computing proximity (cached)                          ')

    scenario_data_path = path.join(PLOT_DATA_PATH, 'scenario_' + user_file)
    if not path.exists(scenario_data_path):
        print(
            f'    {user_file} - Computing scenario                                ')
        ticker = Ticker(len(unfamiliarity_over_time), accuracy=1)
        scenario_over_time = dict()
        old_state = None
        activated = [False, False, False]
        for key in unfamiliarity_over_time.keys():
            if min(alpha_over_time.keys()) < key and key < max(alpha_over_time.keys()):
                closest = min(alpha_over_time.keys(),
                              key=lambda x: abs(key - x))
                alpha = alpha_over_time[closest]
            else:
                alpha = 3

            privacy = privacy_over_time[key]
            unfamiliarity = unfamiliarity_over_time[key]

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
                    activated[item[0]] = item[1]

            scenario_over_time[key] = activated[:]
            old_state = new_state
            ticker.tick('      ')
        pickle.dump(scenario_over_time, open(scenario_data_path, 'wb'))
    else:
        scenario_over_time = pickle.load(open(scenario_data_path, 'rb'))
        print(
            f'    {user_file} - Computing scenario (cached)                          ')


def generate_graph_for_user(user_file, verbose=True):
    if user_file in BLACKLIST:
        plt.clf()
        print(f'    {user_file} - Blacklisted                                ')
        return

    if not path.exists(PLOT_DATA_PATH):
        os.mkdir(PLOT_DATA_PATH)

    alpha_data_path = path.join(PLOT_DATA_PATH, 'alpha_' + user_file)
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    proximity_data_path = path.join(PLOT_DATA_PATH, 'proximity_' + user_file)
    scenario_data_path = path.join(PLOT_DATA_PATH, 'scenario_' + user_file)

    print(f'  Generating graph from file {user_file}')
    print(f'    {user_file} - Loading files                                      ')
    alpha_over_time = pickle.load(open(alpha_data_path, 'rb'))
    privacy_over_time, unfamiliarity_over_time, _, highlight_over_time = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )
    proximity_over_time = pickle.load(open(proximity_data_path, 'rb'))
    scenario_over_time = pickle.load(open(scenario_data_path, 'rb'))

    print(f'    {user_file} - Plotting data                                ')
    plt.gcf().set_size_inches(75, 20)
    plt.gcf().set_dpi(200.0)
    scenario_ax = plt.subplot(6, 1, 1)
    scenario_ax.set_ylabel('Scenario')
    alpha_ax = plt.subplot(6, 1, 2)
    alpha_ax.set_ylabel('Context Familiarity')
    privacy_ax = plt.subplot(6, 1, 3)
    privacy_ax.set_ylabel('Privacy')
    unfamiliarity_ax = plt.subplot(6, 1, 4)
    unfamiliarity_ax.set_ylabel('Unfamiliarity')
    proximity_ax = plt.subplot(6, 1, 5)
    proximity_ax.set_ylabel('Proximity')
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

    unfamiliarity_over_time_keys = list(unfamiliarity_over_time.keys())
    chunks = list()
    start = min_timestamp
    for i in range(len(unfamiliarity_over_time) - 1):
        if unfamiliarity_over_time_keys[i] < min_timestamp:
            continue

        if unfamiliarity_over_time_keys[i + 1] >= max_timestamp:
            break

        if unfamiliarity_over_time_keys[i + 1] - unfamiliarity_over_time_keys[i] > WEEK:
            chunks.append((start, unfamiliarity_over_time_keys[i]))
            start = unfamiliarity_over_time_keys[i + 1]

    chunks.append((start, max_timestamp))

    largest = chunks[0]
    for chunk in chunks:
        if chunk[1] - chunk[0] > largest[1] - largest[0]:
            largest = chunk

    min_timestamp, max_timestamp = largest

    for ax in [scenario_ax, alpha_ax, privacy_ax, unfamiliarity_ax, proximity_ax, actions_ax]:
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

    if path.exists(path.join(VISITS_PATH, user_file)):
        print(
            f'    {user_file} - Plotting visit data                                ')
        visit_data = pickle.load(open(path.join(VISITS_PATH, user_file), 'rb'))

        for visit in visit_data:
            if visit[2] == 'unknown':
                continue

            color = 'green' if visit[2] == 'safe' else 'red'

            scenario_ax.fill_between(
                [visit[0], visit[1]], 1, 0.5, color=color, alpha=0.25)
    else:
        print(
            f'    {user_file} - Plotting visit data (Skipped)                                ')

    print(f'    {user_file} - Plotting auth data                                ')
    auth_data = pickle.load(open(path.join(AUTH_PATH, user_file), 'rb'))
    auth_results = dict()
    for auth_iteration in auth_data.keys():
        color = 'red'
        if min(scenario_over_time.keys()) < auth_iteration and auth_iteration < max(scenario_over_time.keys()):
            closest = min(scenario_over_time.keys(),
                          key=lambda x: abs(auth_iteration - x))
            scenario = scenario_over_time[closest]
            if not scenario[0]:
                color = 'green'

        auth_results[auth_iteration] = (color == 'green')
        scenario_ax.axvline(auth_iteration, 0, .5, color=color)

    pickle.dump(auth_results, open(
        path.join(AUTH_PATH, 'computed_' + user_file), 'wb'))

    print(f'    {user_file} - Plotting values                                ')
    alpha_ax.plot(list(alpha_over_time.keys()), list(
        alpha_over_time.values()), color='purple')
    privacy_ax.plot(list(unfamiliarity_over_time.keys()),
                    list(privacy_over_time.values()), color='blue')
    unfamiliarity_ax.plot(list(privacy_over_time.keys()), list(
        unfamiliarity_over_time.values()), color='green')
    proximity_ax.plot(list(proximity_over_time.keys()), list(
        proximity_over_time.values()), color='gold')

    print(
        f'    {user_file} - Highlighting unfamiliarity/privacy                                ')
    start = -1
    for key, value in highlight_over_time.items():
        if value and start < 0:
            start = key
        elif not value and 0 < start:
            privacy_ax.fill_between(
                [start, key], 1, color='orange', alpha=0.25)
            unfamiliarity_ax.fill_between(
                [start, key], 1, color='orange', alpha=0.25)
            start = -1

    print(f'    {user_file} - Plotting actions                                ')
    action_colors = ['cyan', 'darkblue', 'lime']
    start = [-1, -1, -1]
    current_state = [0, 0, 0]
    for key, value in scenario_over_time.items():
        for index, module in enumerate(value):
            ylims = (
                1 - (index / len(value)),
                (1 - (index + 1) / len(value))
            )
            color = action_colors[index]
            alpha = current_state[index] * .33

            if module and start[index] < 0:
                start[index] = key
                current_state[index] = module

            elif module and module != current_state[index]:
                actions_ax.fill_between(
                    [start[index], key], ylims[0], ylims[1], color=color, alpha=alpha)
                start[index] = key
                current_state[index] = module

            elif not module and 0 < start[index]:
                actions_ax.fill_between(
                    [start[index], key], ylims[0], ylims[1], color=color, alpha=alpha)
                start[index] = -1
                current_state[index] = -1

    if not path.exists('plots'):
        os.mkdir('plots')

    print(f'    {user_file} - Wrapping up                            ')
    plt.tight_layout()
    plt.savefig(path.join('plots', path.splitext(user_file)[0] + '.pdf'))
    plt.clf()
    print(f'    {user_file} - Done                                ')


def generate_specific_graphs_for_user(user_file, verbose=True):
    if user_file not in WHITELIST:
        plt.clf()
        print(
            f'    {user_file} - Not Whitelisted                                ')
        return

    if not path.exists(PLOT_DATA_PATH):
        os.mkdir(PLOT_DATA_PATH)

    alpha_data_path = path.join(PLOT_DATA_PATH, 'alpha_' + user_file)
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    proximity_data_path = path.join(PLOT_DATA_PATH, 'proximity_' + user_file)
    scenario_data_path = path.join(PLOT_DATA_PATH, 'scenario_' + user_file)

    print(f'  Generating graph from file {user_file}')
    print(f'    {user_file} - Loading files                                      ')
    alpha_over_time = pickle.load(open(alpha_data_path, 'rb'))
    privacy_over_time, unfamiliarity_over_time, _, highlight_over_time = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )
    proximity_over_time = pickle.load(open(proximity_data_path, 'rb'))
    scenario_over_time = pickle.load(open(scenario_data_path, 'rb'))

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

    unfamiliarity_over_time_keys = list(unfamiliarity_over_time.keys())
    chunks = list()
    start = min_timestamp
    for i in range(len(unfamiliarity_over_time) - 1):
        if unfamiliarity_over_time_keys[i] < min_timestamp:
            continue

        if unfamiliarity_over_time_keys[i + 1] >= max_timestamp:
            break

        if unfamiliarity_over_time_keys[i + 1] - unfamiliarity_over_time_keys[i] > WEEK:
            chunks.append((start, unfamiliarity_over_time_keys[i]))
            start = unfamiliarity_over_time_keys[i + 1]

    chunks.append((start, max_timestamp))

    largest = chunks[0]
    for chunk in chunks:
        if chunk[1] - chunk[0] > largest[1] - largest[0]:
            largest = chunk

    limits = [
        (largest[0], largest[0] + DAY * 50)  # Context Familiarity thing
    ]

    highlights = [x for x in highlight_over_time.keys()
                  if highlight_over_time[x]]
    best_highlight = 0
    best_total = 0
    for highlight1 in highlights:
        total_nearby = 0
        for highlight2 in highlights:
            if highlight1 != highlight2 and abs(highlight1 - highlight2) < DAY * 2:
                total_nearby += 1

        if total_nearby > best_total:
            best_highlight = highlight1
            best_total = total_nearby

    limits.append((best_highlight - DAY * 3, best_highlight + DAY * 3))

    for i, limits in enumerate(limits):
        print(f'    {user_file} - Plotting data                                ')
        plt.gcf().set_size_inches(15, 8)
        plt.gcf().set_dpi(200.0)
        scenario_ax = plt.subplot(6, 1, 1)
        scenario_ax.set_ylabel('Scenario')
        alpha_ax = plt.subplot(6, 1, 2)
        alpha_ax.set_ylabel('Context Familiarity')
        privacy_ax = plt.subplot(6, 1, 3)
        privacy_ax.set_ylabel('Privacy')
        unfamiliarity_ax = plt.subplot(6, 1, 4)
        unfamiliarity_ax.set_ylabel('Unfamiliarity')
        proximity_ax = plt.subplot(6, 1, 5)
        proximity_ax.set_ylabel('Proximity')
        actions_ax = plt.subplot(6, 1, 6)
        actions_ax.set_ylabel('Modules')
        actions_ax.set_xlabel('Time (Days)')

        min_timestamp, max_timestamp = limits
        for ax in [scenario_ax, alpha_ax, privacy_ax, unfamiliarity_ax, proximity_ax, actions_ax]:
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

        if path.exists(path.join(VISITS_PATH, user_file)):
            print(
                f'    {user_file} - Plotting visit data                                ')
            visit_data = pickle.load(
                open(path.join(VISITS_PATH, user_file), 'rb'))

            for visit in visit_data:
                if visit[2] == 'unknown':
                    continue

                color = 'green' if visit[2] == 'safe' else 'red'

                scenario_ax.fill_between(
                    [visit[0], visit[1]], 1, 0.5, color=color, alpha=0.25)
        else:
            print(
                f'    {user_file} - Plotting visit data (Skipped)                                ')

        print(
            f'    {user_file} - Plotting auth data                                ')
        auth_data = pickle.load(open(path.join(AUTH_PATH, user_file), 'rb'))
        auth_results = dict()
        for auth_iteration in auth_data.keys():
            color = 'red'
            if min(scenario_over_time.keys()) < auth_iteration and auth_iteration < max(scenario_over_time.keys()):
                closest = min(scenario_over_time.keys(),
                              key=lambda x: abs(auth_iteration - x))
                scenario = scenario_over_time[closest]
                if not scenario[0]:
                    color = 'green'

            auth_results[auth_iteration] = (color == 'green')
            scenario_ax.axvline(auth_iteration, 0, .5, color=color)

        pickle.dump(auth_results, open(
            path.join(AUTH_PATH, 'computed_' + user_file), 'wb'))

        print(
            f'    {user_file} - Plotting values                                ')
        alpha_ax.plot(list(alpha_over_time.keys()), list(
            alpha_over_time.values()), color='purple')
        privacy_ax.plot(list(unfamiliarity_over_time.keys()),
                        list(privacy_over_time.values()), color='blue')
        unfamiliarity_ax.plot(list(privacy_over_time.keys()), list(
            unfamiliarity_over_time.values()), color='green')
        proximity_ax.plot(list(proximity_over_time.keys()), list(
            proximity_over_time.values()), color='gold')

        print(
            f'    {user_file} - Highlighting unfamiliarity/privacy                                ')
        start = -1
        for key, value in highlight_over_time.items():
            if value and start < 0:
                start = key
            elif not value and 0 < start:
                privacy_ax.fill_between(
                    [start, key], 1, color='orange', alpha=0.25)
                unfamiliarity_ax.fill_between(
                    [start, key], 1, color='orange', alpha=0.25)
                start = -1

        print(
            f'    {user_file} - Plotting actions                                ')
        action_colors = ['cyan', 'darkblue', 'lime']
        start = [-1, -1, -1]
        current_state = [0, 0, 0]
        for key, value in scenario_over_time.items():
            for index, module in enumerate(value):
                ylims = (
                    1 - (index / len(value)),
                    (1 - (index + 1) / len(value))
                )
                color = action_colors[index]
                alpha = current_state[index] * .33

                if module and start[index] < 0:
                    start[index] = key
                    current_state[index] = module

                elif module and module != current_state[index]:
                    actions_ax.fill_between(
                        [start[index], key], ylims[0], ylims[1], color=color, alpha=alpha)
                    start[index] = key
                    current_state[index] = module

                elif not module and 0 < start[index]:
                    actions_ax.fill_between(
                        [start[index], key], ylims[0], ylims[1], color=color, alpha=alpha)
                    start[index] = -1
                    current_state[index] = -1

        if not path.exists('plots'):
            os.mkdir('plots')

        print(f'    {user_file} - Wrapping up                            ')
        plt.tight_layout()
        plt.savefig(f'plots/{path.splitext(user_file)[0]}_specific{i + 1}.pdf')
        plt.clf()
        print(f'    {user_file} - Done                                ')


def generate_stats_for_user(user_file, verbose=True):
    alpha_data_path = path.join(PLOT_DATA_PATH, 'alpha_' + user_file)
    unfamiliarity_data_path = path.join(
        PLOT_DATA_PATH, 'unfamiliarity_' + user_file)
    proximity_data_path = path.join(PLOT_DATA_PATH, 'proximity_' + user_file)
    scenario_data_path = path.join(PLOT_DATA_PATH, 'scenario_' + user_file)

    print(f'  Generating stats from file {user_file}')
    print(f'    {user_file} - Loading files                                      ')
    alpha_over_time = pickle.load(open(alpha_data_path, 'rb'))
    _, unfamiliarity_over_time, _, highlight_over_time = pickle.load(
        open(unfamiliarity_data_path, 'rb')
    )
    proximity_over_time = pickle.load(open(proximity_data_path, 'rb'))
    scenario_over_time = pickle.load(open(scenario_data_path, 'rb'))
    if path.exists(path.join(VISITS_PATH, user_file)):
        visit_data = pickle.load(open(path.join(VISITS_PATH, user_file), 'rb'))
    else:
        visit_data = None

    safe_context_total = 0
    if visit_data is not None:
        for visit in visit_data:
            if visit[2] == 'safe':
                safe_context_total += (visit[1] - visit[0])

    print(f'    {user_file} - Computing statistics                                ')
    if not path.exists('stats'):
        os.mkdir('stats')

    min_timestamp = min(
        min(alpha_over_time.keys()),
        min(unfamiliarity_over_time.keys()),
        min(proximity_over_time.keys())
    )

    max_timestamp = max(
        max(alpha_over_time.keys()),
        max(unfamiliarity_over_time.keys()),
        max(proximity_over_time.keys())
    )

    auth_data = pickle.load(open(path.join(AUTH_PATH, user_file), 'rb'))

    green_auths = 0

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    true_positives_strict = 0
    false_positives_strict = 0
    true_negatives_strict = 0
    false_negatives_strict = 0

    unclass_total = 0
    unclass_positives = 0
    unclass_negatives = 0

    unclass_total_strict = 0
    unclass_positives_strict = 0
    unclass_negatives_strict = 0

    for auth_iteration in auth_data.keys():
        scenario_val = -1
        if min(scenario_over_time.keys()) < auth_iteration and auth_iteration < max(scenario_over_time.keys()):
            closest = min(scenario_over_time.keys(),
                          key=lambda x: abs(auth_iteration - x))
            scenario_val = scenario_over_time[closest][0]
            if not scenario_val:
                green_auths += 1

        hit = False
        hit_strict = False
        if visit_data is not None:
            for visit in visit_data:
                if visit[2] == 'unknown':
                    continue

                if auth_iteration < visit[0] or visit[1] < auth_iteration:
                    continue

                if visit[2] == 'safe' and not scenario_val:
                    true_positives += 1
                    true_positives_strict += 1
                    hit_strict = True
                elif visit[2] != 'safe' and not scenario_val:
                    false_positives += 1
                    false_positives_strict += 1
                    hit_strict = True
                elif visit[2] != 'safe' and scenario_val:
                    true_negatives += 1
                    if scenario_val != 1:
                        true_negatives_strict += 1
                        hit_strict = True
                elif visit[2] == 'safe' and scenario_val:
                    false_negatives += 1
                    if scenario_val != 1:
                        false_negatives_strict += 1
                        hit_strict = True

                hit = True
                break

        if not hit:
            unclass_total += 1
            if not scenario_val:
                unclass_positives += 1
            else:
                unclass_negatives += 1

        if not hit_strict and scenario_val != 1:
            unclass_total_strict += 1
            if not scenario_val:
                unclass_positives_strict += 1
            else:
                unclass_negatives_strict += 1

    start = -1
    total_highlight = 0
    for key, value in highlight_over_time.items():
        if value and start < 0:
            start = key
        elif not value and 0 < start:
            total_highlight += 1
            start = -1

    start = [-1, -1, -1]
    module_total = [0, 0, 0]
    for key, value in scenario_over_time.items():
        for index, module in enumerate(value):
            if module and start[index] < 0:
                start[index] = key
            elif not module and 0 < start[index]:
                module_total[index] += key - start[index]
                start[index] = -1

    with open(f'stats/{path.splitext(user_file)[0]}', 'w') as f:
        total_time = max_timestamp - min_timestamp
        f.write(f'total time\t{total_time}\n')
        proximity_time = max(proximity_over_time.keys()) - \
            min(proximity_over_time.keys())
        f.write(f'total prox time\t{proximity_time}\n')
        f.write(f'total safe context time\t{safe_context_total}\n')
        f.write(f'total auths\t{len(auth_data)}\n')
        f.write(f'total class auths\t{len(auth_data) - unclass_total}\n')
        f.write(f'total unclass auths\t{unclass_total}\n')
        f.write(f'unclass ratio\t{unclass_total / len(auth_data)}\n')
        f.write(f'green auths\t{green_auths}\n')
        f.write(f'true positive auths\t{true_positives}\n')
        f.write(f'false positive auths\t{false_positives}\n')
        f.write(f'true negative auths\t{true_negatives}\n')
        f.write(f'false negative auths\t{false_negatives}\n')
        f.write(f'true positive auths strict\t{true_positives_strict}\n')
        f.write(f'false positive auths strict\t{false_positives_strict}\n')
        f.write(f'true negative auths strict\t{true_negatives_strict}\n')
        f.write(f'false negative auths strict\t{false_negatives_strict}\n')

        if visit_data is not None:
            all_c_pos = false_positives + true_positives
            all_c_pos_strict = false_positives_strict + true_positives_strict
            all_c_neg = false_negatives + true_negatives
            all_c_neg_strict = false_negatives_strict + true_negatives_strict
            all_g_pos = true_positives + false_negatives
            all_g_pos_strict = true_positives_strict + false_negatives_strict
            all_g_neg = true_negatives + false_positives
            all_g_neg_strict = true_negatives_strict + false_positives_strict

            if all_c_pos > 0:
                safe_precision = true_positives / all_c_pos
                f.write(f'safe auth precision\t{safe_precision}\n')
            if all_c_pos_strict > 0:
                safe_precision_strict = true_positives_strict / all_c_pos_strict
                f.write(
                    f'safe auth precision strict\t{safe_precision_strict}\n')

            if all_g_pos > 0:
                safe_recall = true_positives / all_g_pos
                f.write(f'safe auth recall\t{safe_recall}\n')
            if all_g_pos_strict > 0:
                safe_recall_strict = true_positives_strict / all_g_pos_strict
                f.write(f'safe auth recall strict\t{safe_recall_strict}\n')

            if all_g_neg > 0:
                safe_fallout = false_positives / all_g_neg
                f.write(f'safe auth fpr or fallout\t{safe_fallout}\n')
            if all_g_neg_strict > 0:
                safe_fallout_strict = false_positives_strict / all_g_neg_strict
                f.write(
                    f'safe auth fpr or fallout strict\t{safe_fallout_strict}\n')

            if unclass_total > 0:
                safe_unclass_fallout = unclass_positives / unclass_total
                f.write(f'safe unclassified fallout\t{safe_unclass_fallout}\n')
            if unclass_total_strict > 0:
                safe_unclass_fallout_strict = unclass_positives_strict / unclass_total_strict
                f.write(
                    f'safe unclassified fallout strict\t{safe_unclass_fallout_strict}\n')

            if all_g_neg > 0:
                safe_tnr = true_negatives / all_g_neg
                f.write(f'safe auth tnr\t{safe_tnr}\n')

            if all_c_neg > 0:
                unsafe_precision = true_negatives / all_c_neg
                f.write(f'unsafe auth precision\t{unsafe_precision}\n')
            if all_c_neg_strict > 0:
                unsafe_precision_strict = true_negatives_strict / all_c_neg_strict
                f.write(
                    f'unsafe auth precision strict\t{unsafe_precision_strict}\n')

            if all_g_neg > 0:
                unsafe_recall = true_negatives / all_g_neg
                f.write(f'unsafe auth recall\t{unsafe_recall}\n')
            if all_g_neg_strict > 0:
                unsafe_recall_strict = true_negatives_strict / all_g_neg_strict
                f.write(f'unsafe auth recall strict\t{unsafe_recall_strict}\n')

            if all_g_pos > 0:
                unsafe_fallout = false_negatives / all_g_pos
                f.write(f'unsafe auth fpr or fallout\t{unsafe_fallout}\n')
            if all_g_pos_strict > 0:
                unsafe_fallout_strict = false_negatives_strict / all_g_pos_strict
                f.write(
                    f'unsafe auth fpr or fallout strict\t{unsafe_fallout_strict}\n')

            if unclass_total > 0:
                unsafe_unclass_fallout = unclass_negatives / unclass_total
                f.write(
                    f'unsafe unclassified fallout\t{unsafe_unclass_fallout}\n')
            if unclass_total_strict > 0:
                unsafe_unclass_fallout_strict = unclass_negatives_strict / unclass_total_strict
                f.write(
                    f'unsafe unclassified fallout strict\t{unsafe_unclass_fallout_strict}\n')

            if all_g_pos > 0:
                unsafe_tnr = true_positives / all_g_pos
                f.write(f'unsafe auth tnr\t{unsafe_tnr}\n')

        f.write(f'success ratio\t{(green_auths / len(auth_data)):.4f}\n')
        f.write(f'total highlights\t{total_highlight}\n')
        modules = ['auth', 'theft', 'loss']
        for index, total in enumerate(module_total):
            f.write(f'{modules[index]} time\t{total}\n')
            if index == 0:
                f.write(f'{modules[index]} ratio\t{total / total_time}\n')
            else:
                f.write(f'{modules[index]} ratio\t{total / proximity_time}\n')


def build_mdc_graphs():
    targets = ['gps', 'wlan', 'gpswlan', 'bt', 'app']
    if not all([path.exists(path.join(PARSED_PATH, f'{x}.pkl')) for x in targets]):
        print('Building pickle files from csv')
        build_pkl_files()
    else:
        print('Building pickle files from csv (SKIPPED)')

    if not path.exists(path.join(PARSED_PATH, 'accel.pkl')):
        print('Building accel pickle file from csv')
        build_accel_pkl()
    else:
        print('Building accel pickle file from csv (SKIPPED)')

    build_visits_pkls()

    if not path.exists(BT_USER_PATH):
        split_data('bt')
    else:
        print('Splitting bt data (SKIPPED)')

    folders = [os.listdir(x) for x in [BT_USER_PATH,
                                       GPS_USER_PATH, ACCEL_USER_PATH, APP_USER_PATH]]
    files_intersect = set(folders[0])
    for folder in folders:
        files_intersect = files_intersect & set(folder)

    user_files = sorted(list(files_intersect), key=get_file_size)[9:]
    interval = 30 * 60  # 30 minute intervals

    if not all([path.exists(f'{GPS_USER_PATH}/{x}') for x in user_files]):
        split_data('gps')
    else:
        print('Splitting gps data (SKIPPED)')

    if not all([path.exists(f'{GPSWLAN_USER_PATH}/{x}') for x in user_files]):
        split_data('gpswlan')
    else:
        print('Splitting gpswlan data (SKIPPED)')

    if not all([path.exists(f'{WLAN_USER_PATH}/{x}') for x in user_files]):
        split_data('wlan')
    else:
        print('Splitting wlan data (SKIPPED)')

    if not all([path.exists(f'{ACCEL_USER_PATH}/{x}') for x in user_files]):
        split_data('accel')
    else:
        print('Splitting accel data (SKIPPED)')

    if not all([path.exists(f'{APP_USER_PATH}/{x}') for x in user_files]):
        split_data('app')
    else:
        print('Splitting app data (SKIPPED)')

    if not all([path.exists(f'{AUTH_PATH}/{x}') for x in user_files]):
        print('Generating auth data')
        with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
            p.starmap(generate_auth_for_user, [
                      (user_file, False) for user_file in user_files])
    else:
        print('Generating auth data (SKIPPED)')

    if not all([path.exists(f'{ALPHA_PATH}/{x}') for x in user_files]):
        print('Generating alpha data')
        with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
            p.starmap(generate_alpha_for_user, [
                      (user_file, interval, False) for user_file in user_files])
    else:
        print('Generating alpha data (SKIPPED)')

    if not all([path.exists(f'{UNFAMILIARITY_PATH}/{x}') for x in user_files]):
        print('Generating unfamiliarity data')
        with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
            p.starmap(generate_unfamiliarity_for_user, [
                      (user_file, interval, False) for user_file in user_files])
    else:
        print('Generating unfamiliarity data (SKIPPED)')

    if not all([path.exists(f'{PROXIMITY_PATH}/{x}') for x in user_files]):
        print('Generating proximity data')
        with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
            p.starmap(generate_proximity_for_user, [
                      (user_file, interval, False) for user_file in user_files])
    else:
        print('Generating proximity data (SKIPPED)')

    print('Generating plot data')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_plot_data_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Generating graphs')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_graph_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Generating specific graphs')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_specific_graphs_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Generating stats')
    with Pool(min(os.cpu_count() - 2, len(user_files))) as p:
        p.starmap(generate_stats_for_user, [
                  (user_file, False) for user_file in user_files])

    print('Script Complete.')


if __name__ == '__main__':
    build_mdc_graphs()
