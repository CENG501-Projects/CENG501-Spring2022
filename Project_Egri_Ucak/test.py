from models.trainer.dynamically_grown_gan_trainer import DGGANTrainer

if __name__ == "__main__":
    # trainer = DGGANTrainer(pathdb="./cifar10/output/", imagefolderDataset=True)
    trainer = DGGANTrainer(pathdb="./mnist/mnist_png/training", imagefolderDataset=True)
    trainer.train()
