# Copyright (c) 2025
# Licensed under the MIT License.
# See LICENSE file for details.

from abc import ABC, abstractmethod

class Manager(ABC):  

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def undo(self):
        pass