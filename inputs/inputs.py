import csv

from percept.fields.base import Dict
from percept.utils.models import FieldModel
from percept.conf.base import settings
from percept.utils.input import DataFormats
from percept.utils.models import RegistryCategories
from percept.tests.framework import CSVInputTester
from percept.datahandlers.inputs import BaseInput
import os
from itertools import chain

class NFLInput(BaseInput):
    """
    Extends baseinput to read nfl season data csv
    """
    input_format = DataFormats.csv
    help_text = "Example class to load in csv files."

    def read_input(self, directory, has_header=True):
        """
        stream is any reader object that exposes the .read() interface
        for example:
        csv_input = CSVInput()
        csv_input.read_input(open("csvfile.csv"))
        """

        datafiles = [ f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
        all_csv_data = []
        for infile in datafiles:
            stream = open(os.path.join(directory, infile))
            reader = csv.reader(stream)
            csv_data = []
            for (i, row) in enumerate(reader):
                if i==0:
                    if not has_header:
                        csv_data.append([str(i) for i in xrange(0,len(row))])
                    csv_data.append(row + ["Year"])
                else:
                    csv_data.append(row + [infile])
            all_csv_data.append(csv_data)
        self.data = chain.from_iterable(all_csv_data)