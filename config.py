class Config:
    # Train
    epochs = 100
    freeze_bn = False

    # Dataset
    data_dir = 'data'
    batch_size = '128'
    workers = 12

    # Optimizer and Scheduler
    lr = 0.0005
    momentum = 0.9
    decay = 5e-4
    steps = [0.8, 0.9]
    gamma = 0.3

    # Visualization
    print_freq = 100

    # Tools
    work_dir = 'work_dir'


opt = Config()