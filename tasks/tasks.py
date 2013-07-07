from percept.tasks.base import Task
from percept.fields.base import Complex, List
from inputs.inputs import NFLFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import numpy as np
import calendar
import pandas as pd

log = logging.getLogger(__name__)

class CleanupNFLCSV(Task):
    data = Complex()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Convert from direct nfl data to features."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        row_removal_values = ["", "Week", "Year"]
        int_columns = [0,1,3,4,5,6,8,9,11,12,13]
        for r in row_removal_values:
            data = data[data.iloc[:,0]!=r]

        data.iloc[data.iloc[:,0]=="WildCard",0] = 18
        data.iloc[data.iloc[:,0]=="Division",0] = 19
        data.iloc[data.iloc[:,0]=="ConfChamp",0] = 20
        data.iloc[data.iloc[:,0]=="SuperBowl",0] = 21

        data.iloc[data.iloc[:,1]=="",1] = 0
        data.iloc[data.iloc[:,1]=="@",1] = 1
        data.iloc[data.iloc[:,1]=="N",1] = 2

        data.rename(columns={'' : 'Home'}, inplace=True)

        month_map = {v: k for k,v in enumerate(calendar.month_name)}

        day_map = {v: k for k,v in enumerate(calendar.day_abbr)}

        data['DayNum'] = np.asarray([s.split(" ")[1] for s in data.iloc[:,10]])
        data['MonthNum'] = np.asarray([s.split(" ")[0] for s in data.iloc[:,10]])
        for k in month_map.keys():
            data['MonthNum'][data['MonthNum']==k] = month_map[k]

        del data['Date']

        for k in day_map.keys():
            data['Day'][data['Day']==k] = day_map[k]

        for c in int_columns:
            data.iloc[:,c] = data.iloc[:,c].astype(int)

        return data

class GenerateSeasonFeatures(Task):
    data = Complex()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Convert from direct nfl data to features."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        unique_teams = list(set(list(set(data['Winner/tie'])) + list(set(data['Loser/tie']))))
        unique_years = list(set(data['Year']))

        year_stats = []
        for year in unique_years:
            for team in unique_teams:
                sel_data = data.loc[((data["Year"]==year) & ((data["Winner/tie"] == team) | (data["Loser/tie"] == team))),:]
                losses = sel_data[(sel_data["Loser/tie"] == team)]
                losses["TeamYds"] = losses["YdsL"]
                losses["TeamPts"] = losses["PtsL"]
                losses["OppPts"] = losses["PtsW"]
                losses["OppYds"] = losses["YdsW"]
                losses["Opp"] = losses["Winner/tie"]
                losses = losses.sort(['Week'])

                wins =  sel_data[(sel_data["Winner/tie"] == team)]
                wins["TeamYds"] = wins["YdsW"]
                wins["TeamPts"] = wins["PtsW"]
                wins["OppPts"] = wins["PtsL"]
                wins["OppYds"] = wins["YdsL"]
                wins["Opp"] = wins["Loser/tie"]
                wins = wins.sort(['Week'])

                sel_data = pd.concat(losses, wins)
                sel_data = sel_data.sort(['Week'])
                total_losses = losses.shape[0]
                total_wins = wins.shape[0]
                games_played = sel_data.shape[0]

                home = pd.concat([losses[(losses["Home"] == 1)],wins[(wins["Home"] == 0)]])
                home = home.sort(['Week'])
                total_home = home.shape[0]
                away = pd.concat([wins[(wins["Home"] == 1)],losses[(losses["Home"] == 0)]])
                total_away = away.shape
                away = away.sort(['Week'])

                home_stats = self.calc_stats(home)
                away_stats = self.calc_stats(away)
                win_stats = self.calc_stats(wins)
                loss_stats = self.calc_stats(losses)

                meta_df = self.make_df([team, year, total_wins, total_losses, games_played, total_home, total_away], ["team", "year", "total_wins", "total_losses", "games_played", "total_home", "total_away"])
                stat_list = pd.concat( [meta_df, home_stats, away_stats, win_stats, loss_stats], axis=1)

        return data

    def make_df(self, datalist, labels, name_prefix=""):
        df = pd.DataFrame(datalist).T
        labels = [name_prefix + "_" + l for l in labels]
        df.columns = labels
        return df

    def calc_stats(self, df):
        yds = self.calc_indiv_stats(df, "TeamYds", "OppYds")
        pts = self.calc_indiv_stats(df, "TeamPts", "OppPts")
        pts_per_yard = pts[0]/yds[0]
        opp_pts_per_yard = pts[1]/yds[1]
        eff_ratio = opp_pts_per_yard/pts_per_yard
        return yds + pts + [pts_per_yard, opp_pts_per_yard, eff_ratio]

    def calc_indiv_stats(self, df, teamname, oppname, name_prefix = "", recursed = False):
        stat = np.mean(df[teamname])
        opp_stat = np.mean(df[oppname])
        spread = opp_stat - stat
        ratio = opp_stat/stat
        stats = self.make_df([stat, opp_stat, spread, ratio], ["stat", "opp_stat", "spread", "ratio"], name_prefix)
        if not recursed:
            last_3 = self.calc_indiv_stats(df.iloc[-3:], teamname, oppname, name_prefix = "last_3", recursed= True)
            last_5 = self.calc_indiv_stats(df.iloc[-5:], teamname, oppname, name_prefix = "last_5",recursed= True)
            last_10 = self.calc_indiv_stats(df.iloc[-10:], teamname, oppname, name_prefix = "last_10",recursed= True)
            stats = pd.concat([stats, last_3 , last_5, last_10], axis=1)
        return stats
