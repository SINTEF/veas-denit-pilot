import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.datamodules.components import ChunkedTimeSeriesDataModule
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def process_raw_data(source: Path, destination: Path) -> pd.DataFrame:
    # Load the CSV file from Linea
    df = pd.read_csv(source)
    reformatted_data = {}

    # Iterate through each pair of time and value columns
    for i in range(0, len(df.columns), 2):
        time_column = df.iloc[:, i]
        value_column = df.iloc[:, i + 1]
        parameter_name = df.columns[i]
        for time, value in zip(time_column, value_column):
            if time not in reformatted_data:
                reformatted_data[time] = {}
            reformatted_data[time][parameter_name] = value

    final_df = pd.DataFrame.from_dict(reformatted_data, orient="index")

    final_df.reset_index(inplace=True)
    final_df.rename(columns={"index": "Time"}, inplace=True)

    final_df.to_csv(destination, index=False)

    return final_df


def get_input_dataframe(
    source="../../data/veas_denit_pilot.csv", hour_average=True, include_controlled=True
):
    """
    input:
    * source: file location for the .csv file with data
    * hour_average: whether to average all the data over one hour (True/False)
    * incldue_controlled: whether to include the parameters controlled manually (True/False)
    """

    data = pd.read_csv(source, engine="python")
    # data = data[:-100] # we have a lot of nan's in the final data
    data["time"] = pd.to_datetime(data["Time"], dayfirst=True)

    # Output data:
    # Output is nitrate out of the process
    # The tags are found in 'Tagliste Pilot.xlsx'
    output_labels = ["PHA1-PILB2-QI01.Value"]
    output_data = data[output_labels]
    # Rename the data:
    output_data = output_data.rename(columns={"PHA1-PILB2-QI01.Value": "nitrate_out"})
    if hour_average:
        output_data = output_data.rolling(6, win_type=None).sum() / 6
        output_data.insert(0, "time", data["time"])
        output_data = output_data[6:]
    else:
        output_data.insert(0, "time", data["time"])

    # Input data:

    # Nitrate in dentank:
    nitrat_labels = ["PHA1-DEN1-QI11-QT.PHA1-DEN1-QI11-QT VERDI:Value"]
    nitrat_data = data[nitrat_labels]
    nitrat_data = nitrat_data.rename(
        columns={"PHA1-DEN1-QI11-QT.PHA1-DEN1-QI11-QT VERDI:Value": "nitrate_in"}
    )

    # Oxygen:
    oksygen_labels = ["PHA1-DEN1-QI12-QT.PHA1-DEN1-QI12-QT VERDI:Value"]
    oksygen_data = data[oksygen_labels]
    oksygen_data = oksygen_data.rename(
        columns={"PHA1-DEN1-QI12-QT.PHA1-DEN1-QI12-QT VERDI:Value": "oxygen"}
    )

    # Ammonium:
    ammonium_labels = ["PHA1-NIT1-QI11.PHA1-NIT1-QI11 VERDI:Value"]
    ammonium_data = data[ammonium_labels]
    ammonium_data = ammonium_data.rename(
        columns={"PHA1-NIT1-QI11.PHA1-NIT1-QI11 VERDI:Value": "ammonium"}
    )

    # Orto-P:
    ortop_labels = ["PHA2-DEN2-QI03.PHA2-DEN2-QI03 VERDI:Value"]
    ortop_data = data[ortop_labels]
    ortop_data = ortop_data.rename(columns={"PHA2-DEN2-QI03.PHA2-DEN2-QI03 VERDI:Value": "orto-p"})

    # Trykk over filterbunn:
    # filterpressure_labels = ['PHA1-PILB2-PI01.Value', 'PHA1-PILB2-PI02.Value', 'PHA1-PILB2-PI03.Value',
    #                     'PHA1-PILB2-PI04.Value', 'PHA1-PILB2-PI05.Value', 'PHA1-PILB2-PI06.Value',
    #                     'PHA1-PILB2-PI07.Value', 'PHA1-PILB2-PI08.Value']
    filterpressure_labels = ["PHA1-PILB2-PI01.Value", "PHA1-PILB2-PI08.Value"]
    filterpressure_data = data[filterpressure_labels]
    filterpressure_data = filterpressure_data.rename(
        columns={
            "PHA1-PILB2-PI01.Value": "filterpressure_1",
            # 'PHA1-PILB2-PI02.Value' : 'filterpressure_2',
            # 'PHA1-PILB2-PI03.Value' : 'filterpressure_3',
            # 'PHA1-PILB2-PI04.Value' : 'filterpressure_4',
            # 'PHA1-PILB2-PI05.Value' : 'filterpressure_5',
            # 'PHA1-PILB2-PI06.Value' : 'filterpressure_6',
            # 'PHA1-PILB2-PI07.Value' : 'filterpressure_7',
            "PHA1-PILB2-PI08.Value": "filterpressure_8",
        }
    )

    # Temperature:
    temp_labels = ["RIS-RIS-QI01_TT.RIS-RIS-QI01_TT VERDI:Value"]
    temp_data = data[temp_labels]
    temp_data = temp_data.rename(columns={"RIS-RIS-QI01_TT.RIS-RIS-QI01_TT VERDI:Value": "temp"})

    # Turbidity:
    turb_labels = ["PHA1-SED1-QI01-QT.PHA1-SED1-QI01-QT VERDI:Value"]  # ['PHA4-DEN4-QI13.Value']
    turb_data = data[turb_labels]
    turb_data = turb_data.rename(
        columns={"PHA1-SED1-QI01-QT.PHA1-SED1-QI01-QT VERDI:Value": "turb"}
    )

    # Water to the plant:
    tunnelwater_labels = ["PASL-FB04.Value"]
    tunnelwater_data = data[tunnelwater_labels]
    tunnelwater_data = tunnelwater_data.rename(columns={"PASL-FB04.Value": "tunnelwater"})

    if include_controlled:
        # Mengde prosessvann:
        waterflow_labels = ["PHA1-PILB2-FC01_MV.Value"]
        waterflow_data = data[waterflow_labels]
        waterflow_data = waterflow_data.rename(columns={"PHA1-PILB2-FC01_MV.Value": "waterflow"})

        # Methanol added:
        metanol_labels = ["PHA1-PILB2-FB01.Value"]  # , 'PHA1-PILB2-QB01.Value']
        metanol_data = data[metanol_labels]
        metanol_data = metanol_data.rename(columns={"PHA1-PILB2-FB01.Value": "methanol"})
        # 'PHA1-PILB2-QB01.Value' : 'methanol_alt'})

        # Operational: (whether the pilot is operating as normal, e.g. not being washed)
        operational_data = pd.DataFrame(
            {
                # 'operational': waterflow_data['waterflow'].between(3.2, 3.4) & waterflow_data['waterflow'].shift(4).between(3.2, 3.4)
                "operational": waterflow_data["waterflow"]
                .rolling(window=5)
                .apply(lambda x: x.between(3.1, 3.5).all())
                .astype(bool)
            }
        )

    # Collect the input:
    input_labels = (
        nitrat_labels
        + oksygen_labels
        + ammonium_labels
        + ortop_labels
        + filterpressure_labels
        + temp_labels
        + turb_labels
        + tunnelwater_labels
    )
    input_data = pd.concat(
        [
            nitrat_data,
            oksygen_data,
            ammonium_data,
            ortop_data,
            filterpressure_data,
            temp_data,
            turb_data,
            tunnelwater_data,
        ],
        axis=1,
    )
    if include_controlled:
        input_labels += [metanol_labels, waterflow_labels]  # + settpunkt_labels
        input_data = pd.concat(
            [input_data, metanol_data, waterflow_data, operational_data], axis=1
        )

    if hour_average:
        input_data = input_data.rolling(6, win_type=None).sum() / 6
        input_data.insert(0, "time", data["time"])
        input_data = input_data[6:]
    else:
        input_data.insert(0, "time", data["time"])

    return input_data, output_data


class VEASPilotDataModule(ChunkedTimeSeriesDataModule):
    """Example data module for custom dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        *args,  # you can add arguments unique to your dataset here that you use in .setup
        filename: str = "my_data.csv",  # e.g. argument for filename of data file.
        target_shift: Optional[int] = None,
        hour_average: bool = False,
        include_controlled: bool = True,
        non_operational_data: Optional[Dict[str, Any]] = None,
        # any argument listed here is configurable through the yaml config files and the command line.
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None, load_dir: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        :param stage: The pytorch lightning stage to prepare the dataset for
        :param load_dir: The folder from which to load state of datamodule (e.g. fitted scalers
            etc.).
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            df_input, df_output = get_input_dataframe(
                os.path.join(self.hparams.data_dir, self.hparams.filename),
                hour_average=self.hparams.hour_average,
                include_controlled=self.hparams.include_controlled,
            )

            self.data = df_input.merge(df_output)

            # Shift the output 20 minutes forward in time to account for the time the water uses through the filter
            # (approx. 23 minutes):
            self.data["nitrate_out"] = self.data["nitrate_out"].shift(-2)
            self.data = self.data[:-2]

            if self.hparams.get("non_operational_data") is not None:
                if self.hparams.non_operational_data.get("method") == "linear_interpolation":
                    if self.hparams.non_operational_data.get("interpolate_inputs", False):
                        self.data.loc[~self.data["operational"]] = np.nan
                    else:
                        self.data.loc[
                            ~self.data["operational"], self.hparams.data_variables["target"]
                        ] = np.nan
                    self.data = self.data.interpolate(method="linear")
                    self.data = self.data.ffill()
                    self.data = self.data.bfill()
                else:
                    raise NotImplementedError

            # this is handled by configuring future_covariates lags on the model
            if self.hparams.get("target_shift", None) is not None:
                raise NotImplementedError
                # shift target to do now-casting
                self.data[self.hparams.data_variables["target"]] = self.data[
                    self.hparams.data_variables["target"]
                ].shift(self.hparams.target_shift)

            self.data = self.chunk_dataset(self.data)

            self.data["time"] = pd.to_datetime(self.data["time"])
            self.data["time"] = self.data["time"].dt.tz_localize(None)
            self.data = self.data.set_index("time")

            # Finally, call the _finalize_data_processing function from the base class which performs operations such as
            # data splitting and scaling etc.
            self._finalize_setup(load_dir=load_dir)


if __name__ == "__main__":
    import hydra

    import src.utils

    # You can run this script to test if your datamodule sets up without errors.
    #   Note that data_variables.target needs to be defined
    # Then check notebooks/data_explorer.ipynb to inspect if data looks as expected.

    cfg = src.utils.initialize_hydra(
        os.path.join(os.pardir, os.pardir, "configs", "datamodule", "veas_pilot.yaml"),
        overrides_dict=dict(
            datamodule=dict(data_dir=os.path.join("..", "..", "data"))
        ),  # path to data
        print_config=False,
    )

    dm = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")
    dm.setup("fit")
