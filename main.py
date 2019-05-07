# rebuild linear knn and naive bayes so they are return the model so we can use for bagging or wahtever the fuck


import csv
import matplotlib.pyplot as plot
import random
import numpy as np
import math
from collections import Counter
import operator
from csv import DictWriter, DictReader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
headers = ['player_name', 'college', 'draft_year', 'draft_round', 'draft_number',
           'gp', 'pts', 'reb', 'ast', 'net_rating', 'usg_pct', 'ts_pct', 'ast_pct']
linearRegressor = LinearRegression()


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


def draft_converter(num):
    if num > 0 and num <= 5:
        return "TOP 5"
    elif num >= 6 and num <= 14:
        return "LOTTERY"
    elif num >= 15 and num <= 30:
        return "LATE FIRST ROUND"
    elif num >= 31:
        return "SECOND ROUND"
    else:
        return "UNDRAFTED"


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


def results_linear_reg(slope, b, x):
    guess = (slope*x) + b
    return guess


def linear_reg(x_values, y_values):
    ym = np.mean(y_values)
    xm = np.mean(x_values)
    slope = calc_slope(x_values, y_values, xm, ym)
    b = ym - (xm*slope)
    return slope, b


def test_linear(slope, b, test_x, test_y, names):
    acc = 0
    sleepers = []
<<<<<<< HEAD
    for idx, row in enumerate(test):
        if abs(int(predict[idx])-int(row[-1])) <= 7:
            acc += 1
        if int(predict[idx]) > int(row[-1]):
            sleepers.append(row[0])
        print(row[0],  ": Pred=", int(predict[idx]), " Actual=", int(row[-1]))
    print("Accuracy: ", (acc/len(test_x)))
=======
    for idx, row in enumerate(test_x):
        guess = results_linear_reg(slope, b, row)

        if draft_converter(int(guess)) == draft_converter(test_y[idx]):
            acc += 1
        if int(guess) > int(test_y[idx]):
            sleepers.append(row[0])
        print(names[idx],  ": Pred=", draft_converter(int(int(guess))),
              " Actual=", draft_converter(test_y[idx]))
    print("Accuracy: ", (acc/len(test_y)))
>>>>>>> bd084776936d950f5671161aae45115a7742e6d5


def normalize(xs, x):
    return (x - min(xs)) / ((max(xs) - min(xs)) * 1.0)


<<<<<<< HEAD
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
=======
def knn(train_x, test_x, k):
    x = getNeighbors(train_x, len(train_x[0]), test_x, 5)
    return x
>>>>>>> bd084776936d950f5671161aae45115a7742e6d5


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


<<<<<<< HEAD
if __name__ == '__main__':
    players = org_players('all_seasons.csv', headers)
    table = sort_to_table(players)
    print(table[697])
    naive_bayes(table)
=======
def bootstrap(table):
    return [table[random.randint(0, len(table)-1)] for _ in range(len(table))]


def bagging(table):
    train, test = compute_holdout_partitions(table)

    linear_bag = bootstrap(train)
    knn_bag = bootstrap(train)
    bayes_bag = bootstrap(train)


def useful_rows(table, idxs):
    t = []
    for row in table:
        temp = []
        for i in idxs:
            temp.append(row[i])

        t.append(temp)
    return t


def sklearn_linear():
    pass


if __name__ == '__main__':
    players = org_players('all_seasons.csv', headers)
    table = sort_to_table(players)
    print(table[800])
    bagging(table)

    print("LINEAR (NOT SKLEARN)")
    y_values = get_column(table, 9)
    x_values = useful_rows(table, [0, 2])

    xT, xTs, yTrain, yTest = train_test_split(
        x_values, y_values, test_size=1/3, random_state=0)

    xTest_names = useful_rows(xTs, [0])
    xTrain = useful_rows(xT, [1])
    xTest = useful_rows(xTs, [1])

    slope, b = linear_reg(xTrain, yTrain)
    test_linear(slope, b, xTest, yTest, xTest_names)
    # SKLEARN - Linear
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTrain = xTrain.reshape(-1, 1)
    yTrain = yTrain.reshape(-1, 1)

    xTest = np.array(xTest)
    yTest = np.array(yTest)
    xTest = xTest.reshape(-1, 1)
    yTest = yTest.reshape(-1, 1)

    linearRegressor.fit(xTrain, yTrain)
    yPrediction = linearRegressor.predict(xTest)
    accuracy = linearRegressor.score(xTest, yTest)

    plot.scatter(xTrain, yTrain, color='red')
    plot.plot(xTrain, linearRegressor.predict(xTrain), color='blue')
    plot.title('Draft Number vs PPG')
    plot.xlabel('PPG')
    plot.ylabel('Draft')
    plot.show()
    print("\n \n \n \n \n")
    print("KNN (NOT SKLEARN)")

    y_knn = get_column(table, 9)
    x_knn = useful_rows(table, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    xT, xTs, yTrain, yTest = train_test_split(
        x_knn, y_knn, test_size=1/3, random_state=0)

    knn_test_names = useful_rows(xTs, [0])
    # Normalize our X
    for idx, row in enumerate(xT):
        for ind, i in enumerate(row[1:-1]):
            xs = get_column(xT, ind+1)
            xT[idx][ind+1] = normalize(xs, i)
    for idx, row in enumerate(xTs):
        for ind, i in enumerate(row[1:-1]):
            xs = get_column(xTs, ind+1)
            xTs[idx][ind+1] = normalize(xs, i)

    xT = useful_rows(xT, [1,2,3,4,5,6,7,8,9])
    knn_acc = 0
    for idx, row in enumerate(xTs):
        pred = knn(xT, row[1:], 9)
        if draft_converter(pred) == draft_converter(row[-1]):
            knn_acc += 1
        print(knn_test_names[idx], "Prediction: ",
              draft_converter(pred), " Actual: ", draft_converter(row[-1]))
    print("KNN Accuracy: ", knn_acc/len(xTs))
    print('/n /n /n /n /n')
    y_naive = get_column(table, 9)
    x_naive = useful_rows(table, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    xT, xTs, yTrain, yTest = train_test_split(
        x_naive, y_naive, test_size=1/3, random_state=0)

    naive_test_names = useful_rows(xTs, [0])
    xT = useful_rows(xT, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    xTs = useful_rows(xTs, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    summary = summarizeByClass(xT)
    naive_acc = 0
    for idx, row in enumerate(xTs):
        p = predict(summary, row)
        if draft_converter(p) == draft_converter(row[-1]):
            naive_acc += 1
        print(naive_test_names[idx], " Predict: ",
              draft_converter(p), " Actual: ", draft_converter(row[-1]))
    print("Accuracy: ", naive_acc/len(xTs))
>>>>>>> bd084776936d950f5671161aae45115a7742e6d5
