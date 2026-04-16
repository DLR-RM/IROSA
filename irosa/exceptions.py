################################################################################
# Copyright (c) 2025. Markus Knauer, Joao Silverio                            #
# Licensed under the MIT License. See LICENSE file for details.                 #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 2025                                                                   #
# Author: Markus Knauer                                                        #
# E-mail: markus.knauer@dlr.de                                                 #
# Website: https://github.com/DLR-RM/IROSA                                    #
################################################################################

"""Custom exception classes for IROSA."""


class IROSAError(Exception):
    """Base exception class for IROSA."""


class ConfigurationError(IROSAError):
    """Raised when there's a configuration problem."""


class TrajectoryError(IROSAError):
    """Raised when trajectory processing fails."""


class KMPModelError(IROSAError):
    """Raised when KMP model operations fail."""


class DataValidationError(IROSAError):
    """Raised when data validation fails."""


class ToolExecutionError(IROSAError):
    """Raised when tool execution fails."""


class ToolArgsValidationError(IROSAError):
    """Raised when tool argument validation fails."""
