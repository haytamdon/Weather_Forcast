import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(history, metrics = ['mae', 'val_mae']):
    loss = history.history[metrics[0]]
    val_loss = history.history[metrics[1]]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training MAE")
    plt.plot(epochs, val_loss, "b", label="Validation MAE")
    plt.title("Training and validation MAE")
    plt.legend()
    plt.show()