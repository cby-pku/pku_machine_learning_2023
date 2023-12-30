


class exponential_decay():
    def __init__(self, initial_learning_rate=0.001, decay_rate=0.5, decay_epochs=100):
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs

    def __call__(self, epoch):
        return self.initial_learning_rate * self.decay_rate ** (epoch // self.decay_epochs)