import torch
from collections import deque

class EarlyStopping:
    def __init__(self, patience=10, flag_save=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.qu_val_loss = deque(maxlen=patience)
        self.qu_val_score = deque(maxlen=patience)
        self.qu_train_loss = deque(maxlen=patience)
        self.qu_train_score = deque(maxlen=patience)
        self.flag_save = flag_save

    def step(self, acc, loss, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            if self.flag_save:
                self.save_checkpoint(model)
            self.best_epoch = epoch
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if loss >= torch.mean(torch.tensor(self.qu_val_loss)):
                    self.early_stop = True
        else:
            self.best_score = score
            if self.flag_save:
                self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = epoch
        self.qu_val_score.append(score)
        self.qu_val_loss.append(loss)
        return self.early_stop

    def should_stop(self, train_loss, train_score, val_loss, val_score, epoch):
        flag = False
        if epoch < self.patience:
            pass
        else:
            if val_loss > 0:
                if val_loss >= torch.mean(torch.tensor(self.qu_val_loss)):  # and val_score <= np.mean(self.val_score_list)
                    flag = True
            elif train_loss > torch.mean(torch.tensor(self.qu_train_loss)):
                flag = True
        self.qu_train_loss.append(train_loss)
        self.qu_train_score.append(train_score)
        self.qu_val_loss.append(val_loss)
        self.qu_val_score.append(val_score)

        self.early_stop = flag
        return self.early_stop

    def should_save(self, train_loss, train_score, val_loss, val_score):
        if len(self.qu_val_loss) < 1:
            return False
        if train_loss < min(self.qu_train_loss) and val_score > max(self.qu_val_score):
            # if val_loss < min(self.val_loss_list) and val_score > max(self.val_score_list):
            return True
        else:
            return False

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
