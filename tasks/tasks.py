from __future__ import division
from percept.tasks.base import Task
from percept.tasks.train import Train
from percept.fields.base import Complex, List, Dict, Float
from inputs.inputs import NFLFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import numpy as np
import calendar
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
import random
from itertools import chain
from percept.tests.framework import Tester
import os
from percept.conf.base import settings

log = logging.getLogger(__name__)

def make_df(datalist, labels, name_prefix=""):
    df = pd.DataFrame(datalist).T
    if name_prefix!="":
        labels = [name_prefix + "_" + l for l in labels]
    labels = [l.replace(" ", "_").lower() for l in labels]
    df.columns = labels
    df.index = range(df.shape[0])
    return df

class CleanupNFLCSVTester(Tester):
    test_case_format = {'stream' : basestring, 'dataformat' : basestring}

    def preprocess_input(self, **kwargs):
        stream = kwargs.get('stream')
        dataformat = kwargs.get('dataformat')
        inst = self.cls()
        output_format = inst.data_format
        data = self.read_and_reformat(output_format, stream, dataformat)
        return data, inst

    def test(self, **kwargs):
        super(CleanupNFLCSVTester, self).test(**kwargs)
        data, inst = self.preprocess_input(**kwargs)

        inst.train(data, "")
        assert type(inst.data) == pd.core.frame.DataFrame

class GenerateSeasonFeaturesTester(CleanupNFLCSVTester):
    def test(self, **kwargs):
        data, inst = self.preprocess_input(**kwargs)
        cleanup_nfl_csv = CleanupNFLCSV()
        cleanup_nfl_csv.train(data,"")
        data = cleanup_nfl_csv.data
        inst.train(data, "")
        assert type(inst.data) == pd.core.frame.DataFrame

class GenerateSOSFeaturesTester(CleanupNFLCSVTester):
    def test(self, **kwargs):
        data, inst = self.preprocess_input(**kwargs)
        cleanup_nfl_csv = CleanupNFLCSV()
        cleanup_nfl_csv.train(data,"")
        data = cleanup_nfl_csv.data
        generate_season_features = GenerateSeasonFeatures()
        generate_season_features.train(data,"")
        data = generate_season_features.data
        inst.train(data, "")
        assert type(inst.data) == pd.core.frame.DataFrame

class CleanupNFLCSV(Task):
    tester = CleanupNFLCSVTester
    test_cases = [{'stream' : os.path.join(settings.PROJECT_PATH, "data"), 'dataformat' : NFLFormats.multicsv}]
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
    tester = GenerateSeasonFeaturesTester
    test_cases = [{'stream' : os.path.join(settings.PROJECT_PATH, "data"), 'dataformat' : NFLFormats.multicsv}]
    data = Complex()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Generate season features."

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
        team_dict = {v:k for (k,v) in enumerate(list(set(list(set(data['Winner/tie'])) + list(set(data['Loser/tie'])))))}
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

                sel_data = pd.concat([losses, wins])
                sel_data = sel_data.sort(['Week'])
                total_losses = losses.shape[0]
                total_wins = wins.shape[0]
                games_played = sel_data.shape[0]
                home_wins = wins[(wins["Home"] == 0)].shape[0]
                road_wins = wins[(wins["Home"] == 1)].shape[0]

                home = pd.concat([losses[(losses["Home"] == 1)],wins[(wins["Home"] == 0)]])
                home = home.sort(['Week'])
                total_home = home.shape[0]
                away = pd.concat([wins[(wins["Home"] == 1)],losses[(losses["Home"] == 0)]])
                total_away = away.shape[0]
                away = away.sort(['Week'])

                home_stats = self.calc_stats(home, "home")
                away_stats = self.calc_stats(away, "away")
                win_stats = self.calc_stats(wins, "wins")
                loss_stats = self.calc_stats(losses, "losses")
                team_df = self.make_opp_frame(sel_data, unique_teams, team)
                team_num = team_dict[team]

                meta_df = make_df([team.replace(" ", " ").lower(), year, total_wins, total_losses, games_played, total_home, total_away, home_wins, road_wins, team_num], ["team", "year", "total_wins", "total_losses", "games_played", "total_home", "total_away", "home_wins", "road_wins", "team_num"])
                stat_list = pd.concat([meta_df, home_stats, away_stats, win_stats, loss_stats, team_df], axis=1)
                year_stats.append(stat_list)
        summary_frame = pd.concat(year_stats)
        return summary_frame

    def make_opp_frame(self, df, all_team_names, team):
        team_list = [0 for t in all_team_names]
        for (i,t) in enumerate(all_team_names):
            if t==team:
                continue
            elif t in list(df["Winner/tie"]):
                team_list[i] = 2
            elif t in list(df["Loser/tie"]):
                team_list[i] = 1
        team_df = make_df(team_list, all_team_names)
        return team_df

    def calc_stats(self, df, name_prefix=""):
        yds = self.calc_indiv_stats(df, "TeamYds", "OppYds", name_prefix + "_yds")
        pts = self.calc_indiv_stats(df, "TeamPts", "OppPts", name_prefix + "_pts")
        pts_per_yard = pts.iloc[0,0]/yds.iloc[0,0]
        opp_pts_per_yard = pts.iloc[0,1]/yds.iloc[0,1]
        eff_ratio = opp_pts_per_yard/pts_per_yard
        meta_df = make_df([pts_per_yard, opp_pts_per_yard, eff_ratio], [ name_prefix + "_pts_per_yard", name_prefix + "_opp_pts_per_yard", name_prefix + "_eff_ratio"])
        return pd.concat([yds, pts, meta_df], axis=1)

    def calc_indiv_stats(self, df, teamname, oppname, name_prefix = "", recursed = False):
        stat = np.mean(df[teamname])
        opp_stat = np.mean(df[oppname])
        spread = opp_stat - stat
        ratio = opp_stat/stat
        stats = make_df([stat, opp_stat, spread, ratio], ["stat", "opp_stat", "spread", "ratio"], name_prefix)
        if not recursed and df.shape[0]>0:
            last_3 = self.calc_indiv_stats(df.iloc[-min([3, df.shape[0]]):], teamname, oppname, name_prefix = name_prefix + "_last_3", recursed= True)
            last_5 = self.calc_indiv_stats(df.iloc[-min([5, df.shape[0]]):], teamname, oppname, name_prefix = name_prefix + "_last_5",recursed= True)
            last_10 = self.calc_indiv_stats(df.iloc[-min([10, df.shape[0]]):], teamname, oppname, name_prefix = name_prefix + "_last_10",recursed= True)
            stats = pd.concat([stats, last_3 , last_5, last_10], axis=1)
        return stats

class GenerateSOSFeatures(Task):
    tester = GenerateSOSFeaturesTester
    test_cases = [{'stream' : os.path.join(settings.PROJECT_PATH, "data"), 'dataformat' : NFLFormats.multicsv}]
    data = Complex()
    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Generate strength of schedule features."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = self.predict(data)

    def list_mean(self, l):
        if len(l)>0:
            return sum(l)/len(l)
        else:
            return -1

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        unique_teams = [i.replace(" ", "_").lower() for i in list(set(data['team']))]
        sos = {}
        unique_years = list(set(data['year']))
        for year in unique_years:
            for team in unique_teams:
                sel_data = data.loc[(data["year"]==year) & (data["team"] == team),:]
                if sel_data.shape[0]>0:
                    total_wins = sel_data['total_wins'][0]
                    total_losses = sel_data['total_losses'][0]
                    home_wins = sel_data['home_wins'][0]
                    home_losses = sel_data['home_losses'][0]
                    sos.update({team : [total_wins, total_losses, home_wins, home_losses]})

        sos_data = []
        row_count = data.shape[0]
        for i in xrange(0, row_count):
            sel_data = data.iloc[i,:]
            team = sel_data['team']
            year = sel_data['year']
            next_year_wins = data.loc[(data['team']==team) & (data['year']==year+1), 'total_wins']
            if next_year_wins.shape[0] == 0:
                next_year_wins = sel_data['total_wins']
            opp_list = []
            win_list = []
            loss_list = []
            for opp in unique_teams:
                if team.replace(" ", "_").lower()!=opp and sel_data[opp] in [1,2] and opp in sos:
                    opp_list.append(sos[opp])
                    if sel_data[opp]==1:
                        win_list.append(sos[opp])
                    else:
                        loss_list.append(sos[opp])
            opp_stats = self.calc_opp_stats(opp_list, "opp")
            target_frame = make_df([next_year_wins], ["next_year_wins"])
            last_3 = data.loc[(data['team']==team) & (data['year']<year) & (data['year']>year-4),:]
            if last_3.shape[0]>0:
                last_3_row = pd.DataFrame(list(last_3.mean(axis=0))).T
            else:
                last_3_row = pd.DataFrame([0 for l in xrange(0,data.shape[1])]).T
            last_3_row.columns = ["last_3" + str(l) for l in last_3_row.columns]
            sos_row = pd.concat([opp_stats, target_frame, last_3_row], axis=1)
            sos_data.append(sos_row)
        sos_frame = pd.concat(sos_data)
        full_data = pd.concat([data, sos_frame], axis=1)
        full_data.replace([np.inf, -np.inf], -2, inplace=True)
        full_data.replace([np.nan], -1, inplace=True)
        return full_data

    def calc_opp_stats(self, opp_list, name_prefix = ""):
        opp_total_wins = self.list_mean([o[0] for o in opp_list])
        opp_total_losses = self.list_mean([o[1] for o in opp_list])
        opp_home_wins = self.list_mean([o[2] for o in opp_list])
        opp_road_wins = self.list_mean([o[3] for o in opp_list])

        df = make_df([opp_total_wins, opp_total_losses, opp_home_wins, opp_road_wins], ["opp_total_wins", "opp_total_losses", "opp_home_wins", "opp_road_wins"], name_prefix= name_prefix)
        return df

class RandomForestTrain(Train):
    """
    A class to train a random forest
    """
    colnames = List()
    clf = Complex()
    category = RegistryCategories.algorithms
    namespace = get_namespace(__module__)
    algorithm = RandomForestRegressor
    args = {'n_estimators' : 300, 'min_samples_leaf' : 1, 'compute_importances' : True}

    help_text = "Train and predict with Random Forest."

class CrossValidate(Task):
    data = Complex()
    results = Complex()
    error = Float()
    importances = Complex()
    importance = Complex()
    column_names = List()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)
    args = {'nfolds' : 3, 'algo' : RandomForestTrain}

    help_text = "Convert from direct nfl data to features."

    def cross_validate(self, data, non_predictors, **kwargs):
        nfolds = kwargs.get('nfolds', 3)
        algo = kwargs.get('algo')
        seed = kwargs.get('seed', 1)
        data_len = data.shape[0]
        counter = 0
        fold_length = int(math.floor(data_len/nfolds))
        folds = []
        data_seq = list(xrange(0,data_len))
        random.seed(seed)
        random.shuffle(data_seq)

        for fold in xrange(0, nfolds):
            start = counter

            end = counter + fold_length
            if fold == (nfolds-1):
                end = data_len
            folds.append(data_seq[start:end])
            counter += fold_length

        results = []
        data.index = range(data.shape[0])
        self.importances = []
        for (i,fold) in enumerate(folds):
            predict_data = data.iloc[fold,:]
            out_indices = list(chain.from_iterable(folds[:i] + folds[(i + 1):]))
            train_data = data.iloc[out_indices,:]
            alg = algo()
            target = train_data['next_year_wins']
            train_data = train_data[[l for l in list(train_data.columns) if l not in non_predictors]]
            predict_data = predict_data[[l for l in list(predict_data.columns) if l not in non_predictors]]
            clf = alg.train(train_data,target,**algo.args)
            results.append(alg.predict(predict_data))
            self.importances.append(clf.feature_importances_)
        return results, folds

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        non_predictors = [i.replace(" ", "_").lower() for i in list(set(data['team']))] + ["team", "next_year_wins"]
        self.column_names = [l for l in list(data.columns) if l not in non_predictors]
        results, folds = self.cross_validate(data, non_predictors, **kwargs)
        self.gather_results(results, folds, data)

    def gather_results(self, results, folds, data):
        full_results = list(chain.from_iterable(results))
        full_indices = list(chain.from_iterable(folds))
        partial_result_df = make_df([full_results, full_indices], ["result", "index"])
        partial_result_df = partial_result_df.sort(["index"])
        partial_result_df.index = range(partial_result_df.shape[0])
        result_df = pd.concat([partial_result_df, data[['next_year_wins', 'team', 'year', 'total_wins']]], axis=1)
        result_df = result_df[(result_df['next_year_wins']>0) & result_df['total_wins']>0]
        self.results = result_df
        self.calc_error(result_df)
        self.calc_importance(self.importances, self.column_names)

    def calc_error(self, result_df):
        filtered_df = result_df[result_df['year']<np.max(result_df['year'])]
        self.error = np.mean(np.abs(filtered_df['result'] - filtered_df['next_year_wins']))

    def calc_importance(self, importances, col_names):
        importance_frame = pd.DataFrame(importances)
        importance_frame.columns = col_names
        self.importance = importance_frame.mean(axis=0)
        self.importance.sort(0)

    def predict(self, data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """

        pass

class SequentialValidate(CrossValidate):
    args = {'min_years' : 4, 'algo' : RandomForestTrain}
    def sequential_validate(self, data, non_predictors, **kwargs):
        algo = kwargs.get('algo')
        seed = kwargs.get('seed', 1)
        min_years = kwargs.get('min_years', 4)
        random.seed(seed)

        min_year = np.min(data['year'])
        start_year = min_year + min_years

        end_year = np.max(data['year'])
        results = []
        predict_rows = []
        self.importances = []
        for year in xrange(start_year, end_year+1):
            train_data = data[data['year']< year]
            predict_full = data[data['year'] == year]

            alg = algo()

            target = train_data['next_year_wins']
            train_data = train_data[[l for l in list(train_data.columns) if l not in non_predictors]]
            predict_data = predict_full[[l for l in list(predict_full.columns) if l not in non_predictors]]

            clf = alg.train(train_data,target, **algo.args)
            results.append(alg.predict(predict_data))
            predict_rows.append(predict_full)
            self.importances.append(clf.feature_importances_)
        predict_frame = pd.concat(predict_rows)
        predict_frame.index = range(predict_frame.shape[0])
        full_results = list(chain.from_iterable(results))
        predict_frame['result'] = full_results
        result_df = predict_frame[['next_year_wins', 'team', 'year', 'total_wins', 'result']]
        self.results = result_df
        self.calc_error(result_df)
        self.column_names = [l for l in list(data.columns) if l not in non_predictors]
        self.calc_importance(self.importances, self.column_names)

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        non_predictors = [i.replace(" ", "_").lower() for i in list(set(data['team']))] + ["team", "next_year_wins"]
        self.sequential_validate(data, non_predictors, **kwargs)




