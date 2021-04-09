from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks

def get_model(cfg):
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        input_shape=(cfg.model.img_size, cfg.model.img_size, 3),
        pooling="avg"
    )
    features = base_model.output
    num_unit = 101
    if cfg.model.use_age_group:
        num_unit = 9
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=num_unit, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    return model


def get_optimizer(cfg):
    if cfg.train.optimizer_name == "sgd":
        return SGD(lr=cfg.train.lr, momentum=0.9, nesterov=True)
    elif cfg.train.optimizer_name == "adam":
        return Adam(lr=cfg.train.lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def get_scheduler(cfg, initial_epoch):
    class Schedule:
        def __init__(self, nb_epochs, initial_lr):
            self.epochs = nb_epochs
            self.initial_lr = initial_lr

        def __call__(self, epoch_idx):
            total_epoch_idx = epoch_idx + initial_epoch
            print(f"-----------total_epoch_idx: {total_epoch_idx}----------")
            if total_epoch_idx < self.epochs * 0.25:
                return self.initial_lr
            elif total_epoch_idx < self.epochs * 0.50:
                return self.initial_lr * 0.2
            elif total_epoch_idx < self.epochs * 0.75:
                return self.initial_lr * 0.04
            return self.initial_lr * 0.008
    return Schedule(cfg.train.epochs, cfg.train.lr)

def get_logger(checkpoint_dir, initial_epoch, base_lr):
  with open(str(checkpoint_dir) + '/logs.csv', 'a+') as f:
      f.write("epoch,loss,pred_age_accuracy,pred_gender_accuracy,val_loss,val_pred_age_accuracy,val_pred_gender_accuracy,lr")
  class saveLossCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      with open(str(checkpoint_dir) + '/logs.csv', 'a+') as f:
        f.write(','.join([f"{initial_epoch + epoch + 1}",
                         f"{logs.get('loss')}",
                         f"{logs.get('pred_age_accuracy')}",
                         f"{logs.get('pred_gender_accuracy')}",
                         f"{logs.get('val_loss')}",
                         f"{logs.get('val_pred_age_accuracy')}",
                         f"{logs.get('val_pred_gender_accuracy')}",
                         f"{logs.get('lr') or base_lr}\n"]))
  return saveLossCallback()
