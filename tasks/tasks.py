from percept.tasks.base import Task
from percept.fields.base import Complex
from inputs.inputs import NFLFormats
from percept.utils.models import RegistryCategories, get_namespace

class ConvertNFLFeatures(Task):
    data = Complex()

    data_format = NFLFormats.dataframe

    category = RegistryCategories.preprocessors
    namespace = get_namespace(__module__)

    help_text = "Convert from direct nfl data to features."

    def train(self, data, **kwargs):
        """
        Used in the training phase.  Override.
        """
        pass

    def predict(self, test_data, **kwargs):
        """
        Used in the predict phase, after training.  Override
        """
        pass
