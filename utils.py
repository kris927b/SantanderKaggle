import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from keras.callbacks import Callback

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def plot_learning(metrics):
    loss, acc, val_loss, val_acc = metrics['loss'], metrics['acc'], metrics['val_loss'], metrics['val_acc']
    _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(loss, label="Training loss", linewidth=3.0)
    ax2.plot(acc, label="Training accuracy", linewidth=3.0)
    ax1.plot(val_loss, label="Validation loss", linewidth=3.0)
    ax2.plot(val_acc, label="Validation accuracy", linewidth=3.0)
    ax1.set_title("Training Loss VS. Validation Loss", fontsize=24)
    ax2.set_title("Training Accuracy VS. Validation Accuracy", fontsize=24)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.legend(fontsize=16, fancybox=True, framealpha=0)
    ax2.legend(fontsize=16, fancybox=True, framealpha=0)
    plt.show()

def F1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

def auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)