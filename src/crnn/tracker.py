from matplotlib.figure import Figure
from dataclasses import dataclass
from .utils import plot
from .utils import base as utils
from typing import Optional, Literal, List
import os
from datetime import datetime
from .models.base import ConstrainedModule
from .io import save_model, save_sequences_to_mat
import time
from numpy.typing import NDArray
import numpy as np

@dataclass
class Event:
    msg: str
@dataclass
class Log(Event):
    log_msg: str
@dataclass
class SaveFig(Event):
    fig: Figure
    name: str

@dataclass
class Start(Event):
    pass

@dataclass
class Stop(Event):
    pass

@dataclass
class SaveModel(Event):
    model: ConstrainedModule

@dataclass
class SaveSequences(Event):
    e_hats: List[NDArray[np.float64]]
    es: List[NDArray[np.float64]]
    filename: str


class BaseTracker():
    def __init__(self, directory:Optional[str] = os.environ['HOME'], model_name: str = '', type:Literal['training','validation']='training') -> None:
        self.directory = directory
        self.model_name = model_name
        self.log_file_path = os.path.join(self.directory, f'{type}.log')
    
    def track(self, event:Event) -> None:
        if isinstance(event, Log):
            print(event.log_msg)
            self.write_to_logfile(event.log_msg)
        elif isinstance(event, SaveFig):
            plot.save_fig(event.fig, event.name, self.directory)
            self.write_to_logfile(f'save fig {event.name} in {self.directory}')
        elif isinstance(event, Start):
            self.start_time = time.time()
            self.write_to_logfile(f'--- Start model {self.model_name} ---')
        elif isinstance(event, Stop):
            self.write_to_logfile(f'--- Stop duration: {utils.get_duration_str(self.start_time,time.time())} ---')
        elif isinstance(event, SaveModel):
            save_model(event.model,self.directory,self.get_model_filename())
            self.write_to_logfile(f'Save model to {self.get_model_filename()} in {self.directory}')
        elif isinstance(event, SaveSequences):
            save_sequences_to_mat(event.e_hats, event.es, os.path.join(self.directory,event.filename))
            self.write_to_logfile(f'Save sequences to {event.filename} in {self.directory}')
        else:
            raise ValueError(f"Event is not defined")


    def write_to_logfile(self, msg: str) -> None:
        with open(self.log_file_path, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} - {msg}\n")

    def get_model_filename(self) -> str:
        return f'parameters-{self.model_name}.pth'




        