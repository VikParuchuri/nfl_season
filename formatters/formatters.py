from pandas import DataFrame
import numpy as np
from percept.utils.models import FieldModel

from percept.fields.base import Dict
from percept.conf.base import settings
from percept.utils.models import RegistryCategories, get_namespace
from percept.utils.input import DataFormats
from percept.tests.framework import JSONFormatTester
from percept.datahandlers.formatters import BaseFormat, JSONFormat
from inputs.inputs import NFLFormats
import os
import re
import logging
log = logging.getLogger(__name__)

class NFLFormatter(JSONFormat):
    test_cases = [
        {'stream' : os.path.abspath(os.path.join(settings.PACKAGE_PATH,'tests/data/csv/1/data.csv')), 'dataformat' : DataFormats.csv},
        {'stream' : os.path.abspath(os.path.join(settings.PROJECT_PATH, "data")), 'dataformat' : NFLFormats.multicsv}
    ]
    namespace = get_namespace(__module__)

    def from_multicsv(self,input_data):
        """
        Reads multicsv format input data and converts to json.
        """
        reformatted_data = []
        for (i,row) in enumerate(input_data):
            if i==0:
                headers = row
            else:
                data_row = {}
                for (j,h) in enumerate(headers):
                    if j in row:
                        data_row.update({h : row[j]})
                    else:
                        data_row.update({h : 0})
                reformatted_data.append(data_row)
        return reformatted_data

