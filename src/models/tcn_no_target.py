import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from darts.logging import raise_if_not
from darts.models.forecasting.pl_forecasting_module import io_processor
from darts.models.forecasting.tcn_model import TCNModel, _TCNModule

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class _TCNNoTargetModule(_TCNModule):
    """This class allows to create custom RNN modules that can later be used with Darts'
    `RNNModel`. It adds the backbone that is required to be used with Darts'
    `TorchForecastingModel` and `RNNModel`.

    To create a new module, subclass from `CustomRNNModule` and:

    * Define the architecture in the module constructor (`__init__()`)

    * Add the `forward()` method and define the logic of your module's forward pass

    * Use the custom module class when creating a new `RNNModel` with parameter `model`.

    You can use `darts.models.forecasting.rnn_model._RNNModule` as an example.

    Parameters
    ----------
    input_size
        The dimensionality of the input time series.
    hidden_dim
        The number of features in the hidden state `h` of the RNN module.
    num_layers
        The number of recurrent layers.
    target_size
        The dimensionality of the output time series.
    nr_params
        The number of parameters of the likelihood (or 1 if no likelihood is used).
    dropout
        The fraction of neurons that are dropped in all-but-last RNN layers.
    **kwargs
        all parameters required for `darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
        base class.
    """

    def _produce_train_output(self, input_batch: Tuple):
        """Feeds PastCovariatesTorchModel with input and output chunks of a
        PastCovariatesSequentialDataset for training.

        Parameters:
        ----------
        input_batch
            ``(past_target, past_covariates, static_covariates)``
        """
        past_target, past_covariates, static_covariates = input_batch
        # Currently all our PastCovariates models require past target and covariates concatenated
        inpt = (
            (past_covariates),
            static_covariates,
        )
        return self(inpt)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """Feeds PastCovariatesTorchModel with input and output chunks of a
        PastCovariatesSequentialDataset to forecast the next ``n`` target values per target
        variable.

        Parameters:
        ----------
        n
            prediction length
        input_batch
            ``(past_target, past_covariates, future_past_covariates, static_covariates)``
        roll_size
            roll input arrays after every sequence by ``roll_size``. Initially, ``roll_size`` is equivalent to
            ``self.output_chunk_length``
        """
        dim_component = 2
        (
            past_target,
            past_covariates,
            future_past_covariates,
            static_covariates,
        ) = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = past_covariates.shape[dim_component] if past_covariates is not None else 0

        # ------ changed ------ #
        input_past = past_covariates
        # ------ changed ------ #

        out = self._produce_predict_output(x=(input_past, static_covariates))[
            :, self.first_prediction_index :, :
        ]

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = prediction_length + self.output_chunk_length - n
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]

            # ==========> PAST INPUT <==========
            # roll over input series to contain the latest target and covariates
            input_past = torch.roll(input_past, -roll_size, 1)

            # ------ changed ------ #
            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length :, :]
            # ------ changed ------ #

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = (
                    prediction_length - self.input_chunk_length,
                    prediction_length,
                )

            # update past covariates to include next `roll_size` future past covariates elements
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :] = future_past_covariates[:, left_past:right_past, :]
            elif n_past_covs:
                input_past[:, :, :] = future_past_covariates[:, left_past:right_past, :]

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(x=(input_past, static_covariates))[
                :, self.first_prediction_index :, :
            ]

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction


class TCNNoTargetModel(TCNModel):
    """Version of TCNModel that does not use the target as input, only the past_covariates."""

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        """Create the model."""
        raise_if_not(
            self.uses_past_covariates,
            "TCNNoTargetModel does not use target variable, and as such must use past_covariates",
            log,
        )

        raise_if_not(
            train_sample[1] is not None and train_sample[1].shape[1] > 0,
            "TCNNoTargetModel does not use target variable, and as such must use past_covariates",
            log,
        )

        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[1].shape[1]
        output_dim = train_sample[-1].shape[1]
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TCNNoTargetModule(
            input_size=input_dim,
            target_size=output_dim,
            nr_params=nr_params,
            kernel_size=self.kernel_size,
            num_filters=self.num_filters,
            num_layers=self.num_layers,
            dilation_base=self.dilation_base,
            target_length=self.output_chunk_length,
            dropout=self.dropout,
            weight_norm=self.weight_norm,
            **self.pl_module_params,
        )


setattr(sys.modules[TCNModel.__module__], "TCNNoTargetModel", TCNNoTargetModel)
setattr(sys.modules[_TCNModule.__module__], "_TCNNoTargetModule", _TCNNoTargetModule)
