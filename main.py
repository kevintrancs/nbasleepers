import csv
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


if __name__ == '__main__':
    players = org_players('all_seasons.csv', headers)
