class Config:
    # Train
    epochs = 20
    freeze_bn = False

    # Dataset
    data_dir = 'data'
    batch_size = 192
    input_size = (224, 224)
    workers = 2

    # Optimizer and Scheduler
    lr = 0.05
    momentum = 0.9
    decay = 5e-4
    steps = [0.8, 0.9]
    gamma = 0.3

    # Visualization
    print_freq = 100

    # Tools
    work_dir = 'work_dir'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
