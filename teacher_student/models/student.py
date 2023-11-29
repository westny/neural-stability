from models.neuralode import NeuralODE


class Student(NeuralODE):
    def __init__(self, config: dict):
        super().__init__(config['n_states'],
                         config['n_inputs'],
                         config['n_hidden'],
                         config['n_layers'],
                         config['ode_solver'],
                         config['step_size'],
                         1.0,
                         False,
                         config['stability_init'],
                         config['complex_poles'],
                         config['activation'])
