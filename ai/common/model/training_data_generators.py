class TrainingDataGenerators:
    """
    Training Data Generators Data Class
    """
    def __init__(self, training_data_generator, validation_data_generator, testing_data_generatr):
        self.training_data_generator = training_data_generator
        self.validation_data_generator = validation_data_generator
        self.testing_data_generator = testing_data_generatr
