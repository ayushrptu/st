from .csv_data import CSVData, CSVsData, LabeledCSVData
from .toy_data import ToyData, gp_likelihood, create_toydata, test_toydata, FinToyData
from .data_transforms import Mask_TS, Gaussian_Noise
from .data_tools import Smalldata
from .speech import SpeechData

# List of available datasets
data_dict = {
    "CSV": CSVData,
    "CSVs": CSVsData,
    "LabeledCSV": LabeledCSVData,
    "GP": ToyData,
    "TIMIT": SpeechData,
    "FIN": FinToyData
}

# List of available transforms
transform_dict = {
    "masking": Mask_TS,
    "gaussian": Gaussian_Noise
}
