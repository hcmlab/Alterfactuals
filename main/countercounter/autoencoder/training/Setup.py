class Setup:

    def __init__(self):
        self.model = None
        self.scaled_model = None

        self.criterion = None
        self.optimizer = None
        self.ssim = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.epochs = None

        self.checkpoints_dir = None
        self.root_dir = None
        self.config_nr = None
        self.tensorboard_dir = None
        self.model_dir = None

        self.ssim_function = None