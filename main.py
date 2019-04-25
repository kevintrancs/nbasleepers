import csv
import random
import numpy as np
import math
from collections import Counter
import operator
from csv import DictWriter, DictReader
headers = ['player_name', 'college', 'draft_year', 'draft_round', 'draft_number',
           'gp', 'pts', 'reb', 'ast', 'net_rating', 'usg_pct', 'ts_pct', 'ast_pct']


class Player:
    def __init__(self, data):
        self.player_name = data['player_name']
        self.college = data['college']
        self.draft_year = data['draft_year']
        self.draft_round = data['draft_round']
        self.draft_number = data['draft_number']
        self.gp = [float(data['gp'])]
        self.pts = [float(data['pts'])]
        self.reb = [float(data['reb'])]
        self.ast = [float(data['ast'])]
        self.net_rating = [float(data['net_rating'])]
        self.usg_pct = [float(data['usg_pct'])]
        self.ts_pct = [float(data['ts_pct'])]
        self.ast_pct = [float(data['ast_pct'])]


def org_players(file, headers):
    with open(file) as read:
        csv_reader = csv.DictReader(read)
        players = {}
        for row in csv_reader:
            if row['player_name'] not in players:
                data = {}
                for i in headers:
                    data[i] = row[i]
                players[row['player_name']] = (Player(data))
            else:
                for i in headers[5:len(headers)]:
                    getattr(players[row['player_name']],
                            i).append(float(row[i]))

        for k, v in players.items():
            generate_career_avgs(v)
        return players


def generate_career_avgs(player):
    player.gp = sum(player.gp)
    player.pts = sum(player.pts) / len(player.pts)
    player.reb = sum(player.reb) / len(player.reb)
    player.ast = sum(player.ast) / len(player.ast)
    player.net_rating = sum(player.net_rating) / len(player.net_rating)
    player.usg_pct = sum(player.usg_pct) / len(player.usg_pct)
    player.ts_pct = sum(player.ts_pct) / len(player.ts_pct)
    player.ast_pct = sum(player.ast_pct) / len(player.ast_pct)
    if player.draft_number == "Undrafted":  # set all undrafted to 0
        player.draft_year = 0.0
        player.draft_round = 0.0
        player.draft_number = 0.0


def sort_to_table(players):
    table = []
    for k, v in players.items():
        table.append([v.player_name, v.gp, v.pts, v.reb,
                      v.ast, v.net_rating, v.usg_pct, v.ts_pct, v.ast_pct, int(v.draft_number)])
    return table

# 0 = name, 1 = draftNumber (classifier)


def get_column(table, index):
    col = []
    for row in table:
        col.append(float(row[index]))
    return col


def compute_holdout_partitions(table):
    randomized = table[:]
    n = len(randomized)
    for i in range(n):
        rand_index = random.randrange(0, n)
        randomized[i], randomized[rand_index] = randomized[rand_index], randomized[i]
    split_index = int(2 / 3 * n)
    train_set = randomized[:split_index]
    test_set = randomized[split_index:]
    return train_set, test_set


def calc_slope(X, Y, mean_x, mean_y):
    numer = 0
    denom = 0
    m = len(X)
    for i in range(m):
        numer += (X[i] - mean_x) * (Y[i] - mean_y)
        denom += (X[i] - mean_x) ** 2
    slope = numer / denom
    return slope


def predictor(x, m, b):
    y_guess = []
    for i in x:
        y_guess.append((m*i) + b)

    return y_guess


def linear_reg(table, x_ind, y_ind):
    train, test = compute_holdout_partitions(table)

    y_values = get_column(train, y_ind)
    x_values = get_column(train, x_ind)

    ym = np.mean(y_values)
    xm = np.mean(x_values)

    slope = calc_slope(x_values, y_values, xm, ym)
    b = ym - (xm*slope)

    test_x = get_column(test, x_ind)
    test_y = get_column(test, y_ind)
    predict = predictor(test_x, slope, b)
    acc = 0
    sleepers = []
    for idx, row in enumerate(test):
        if abs(int(predict[idx])-int(row[-1])) <= 7:
            acc += 1
        if int(predict[idx]) > int(row[-1]):
            sleepers.append(row[0])
        print(row[0],  ": Pred=", int(predict[idx]), " Actual=", int(row[-1]))
    print("Accuracy: ", (acc/len(test_x)))


def normalize(xs, x):
    return (x - min(xs)) / ((max(xs) - min(xs)) * 1.0)


def knn(table, k):

    train, test = compute_holdout_partitions(table)

    knn_train = []
    knn_test = []
    knn_test_names = []

    for row in test:
        knn_test_names.append(row[0])
        knn_test.append(row[1::])

    for row in train:
        knn_train.append(row[1::])

    for idx, row in enumerate(knn_train):
        for ind, i in enumerate(row[:-1]):
            xs = get_column(knn_train, ind)
            knn_train[idx][ind] = normalize(xs, i)

    for idx, row in enumerate(knn_test):
        for ind, i in enumerate(row[:-1]):
            xs = get_column(knn_test, ind)
            knn_test[idx][ind] = normalize(xs, i)

    for ind, i in enumerate(knn_test):
        x = getNeighbors(knn_train, len(knn_test[0]), knn_test[ind], 3)
        print(knn_test_names[ind], " Predicted label: ",
              int(x), " Actual: ", knn_test[ind][-1])


def compute_distance(v1, v2, length):
    distance = 0
    for x in range(length):
        distance += pow((v1[x] - v2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(training_set, n, instance, k):
    row_distances = []
    for row in training_set:
        d = compute_distance(row, instance, n - 1)
        row_distances.append([d, row])
    row_distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for x in range(k):
        neighbors.append(row_distances[x][1])

    lst = []
    for ind, x in enumerate(neighbors):
        lst.append(neighbors[ind][-1])
    data = Counter(lst)
    return data.most_common(1)[0][0]


def gaussian(x, mean, sdev):
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second


def sep_class(table):
    label = {}
    for idx, row in enumerate(table):
        if row[-1] not in label:
            label[row[-1]] = []
        label[row[-1]].append(row)
    return label


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(np.mean(attribute), np.std(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = sep_class(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def naive_bayes(table):
    test_names = []
    train_x = []
    test_x = []
    train, test = compute_holdout_partitions(table)
    for row in train:
        train_x.append(row[1::])

    for row in test:
        test_names.append(row[0])
        test_x.append(row[1::])

    summary = summarizeByClass(train_x)
    acc = 0
    for idx, row in enumerate(test_x):
        p = predict(summary, row)
        if abs(p - row[-1]) <= 7:
            acc += 1
        print(test_names[idx], " Predict: ", p, " Actual: ", row[-1])
    print("Accuracy: ", acc/len(test_x))


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= gaussian(x, mean, stdev)
    return probabilities


if __name__ == '__main__':
    players = org_players('all_seasons.csv', headers)
    table = sort_to_table(players)
    print(table[697])
    naive_bayes(table)