from percept.tasks.base import Task
from percept.fields.base import Complex, List
from inputs.inputs import NFLFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import numpy as np
import calendar

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

