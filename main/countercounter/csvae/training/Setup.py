class Setup:

    def __init__(self):
        self.classifier = None

        self.csvae = None

        self.encoder_x_to_z = None
        self.encoder_xy_to_w = None
        self.encoder_y_to_w = None
        self.decoder_zw_to_x = None
        self.decoder_z_to_y = None

        self.main_optimizer = None
        self.delta_optimizer = None

        self.main_scheduler = None
        self.delta_scheduler = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.epochs = None

        self.checkpoints_dir = None
        self.root_dir = None
        self.config_nr = None
        self.tensorboard_dir = None
        self.model_dir = None
        self.image_sample_size = None
        self.image_dir = None

        self.beta1 = None
        self.beta2 = None
        self.beta3 = None
        self.beta4 = None
        self.beta5 = None

        self.x_recon_loss_function = None

        self.z_size = None