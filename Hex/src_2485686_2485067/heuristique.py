# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from abc import ABC, abstractmethod

class Heuristique(ABC):  

    @abstractmethod
    def execute(self):
        pass