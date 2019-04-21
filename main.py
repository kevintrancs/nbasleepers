import csv
import random
import numpy as np
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
        table.append([v.player_name, v.draft_number, v.gp, v.pts, v.reb,
                      v.ast, v.net_rating, v.usg_pct, v.ts_pct, v.ast_pct])
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
        if abs(int(predict[idx])-int(row[1])) <= 7:
            acc += 1
        if int(predict[idx]) > int(row[1]):
            sleepers.append(row[0])
        print(row[0],  ": Pred=", int(predict[idx]), " Actual=", int(row[1]))
    print("Accuracy: ", (acc/len(test_x)))
    print("Sleeps", sleepers)


def normalize(xs, x):
    return (x - min(xs)) / ((max(xs) - min(xs)) * 1.0)


if __name__ == '__main__':
    players = org_players('all_seasons.csv', headers)
    table = sort_to_table(players)
