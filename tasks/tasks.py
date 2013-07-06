from percept.tasks.base import Task
from percept.fields.base import Complex, List
from inputs.inputs import NFLFormats
from percept.utils.models import RegistryCategories, get_namespace
import logging
import numpy as np

log = logging.getLogger(__name__)

class ConvertNFLFeatures(Task):
    data = Complex()
    target = Complex()
    feature_names = List()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Convert from direct nfl data to features."

    def train(self, data, target, **kwargs):
        """
        Used in the training phase.  Override.
        """
        self.data = data
        self.target = target
        self.feature_names = ["hello", "bye"]

    def predict(self, test_data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        pass
