class Setup:
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.plausibility_discriminator = None

        self.generator_loss = None
        self.plausibility_generator_loss = None
        self.discriminator_loss = None
        self.discriminator_loss_plausibility = None

        self.generator_loss_calculator = None
        self.discriminator_loss_calculator = None

        self.loss_printer = None
        self.loss_tensorboard_logger = None

        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.discriminator_plausibility_optimizer = None

        self.classifier = None
        self.autoencoder = None
        self.kde = None
        self.csae = None
        self.svm = None

        self.ssim = None

        self.train_loader = None
        self.val_loader = None

        self.full_set_loader = None
        self.test_loader = None

        self.lambda_classifier = None
        self.lambda_ssim = None
        self.lambda_autoencoder = None
        self.lambda_kde = None
        self.lambda_plausibility_discriminator = None
        self.lambda_csae = None
        self.lambda_svm = None

        self.epochs = None

        self.checkpoints_dir = None
        self.root_dir = None
        self.config_nr = None
        self.tensorboard_dir = None
        self.model_dir = None
        self.image_dir = None

        self.image_sample_size = None

        self.minimal_logging = None

        self.one_sided_label_smoothing = None
        self.one_sided_label_smoothing_plausibility = None

        self.pass_labels_to_discriminator = None
        self.discriminator_input_size = None
        self.same_class = None

        self.ssim_function = None

        self.weight_clipping = None
        self.clipping = None