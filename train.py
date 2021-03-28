from pathlib import Path
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from src.factory import get_model, get_optimizer, get_scheduler
from src.generator import ImageSequence

checkpoint_dir = Path(to_absolute_path(__file__)).parent.parent.joinpath('drive', 'MyDrive', 'AgeGenderCheckpoint')
checkpoint_dir.mkdir(exist_ok=True)

class saveLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        with open(str(checkpoint_dir) + '/logs.txt', 'a+') as f:
            f.write(f"{epoch},{logs.get('val_loss')},{logs.get('loss')}\n")

@hydra.main(config_path="src/config.yaml")
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks = [WandbCallback()]
    else:
        callbacks = []
    weight_file = cfg.train.weight_file

    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"{cfg.data.db}.csv")
    df = pd.read_csv(str(csv_path))
    train, val = train_test_split(df, random_state=42, test_size=0.2)
    train_gen = ImageSequence(cfg, train, "train")
    val_gen = ImageSequence(cfg, val, "val")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_model(cfg)
        opt = get_optimizer(cfg)
        scheduler = get_scheduler(cfg)
        model.compile(optimizer=opt,
                      loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
                      metrics=['accuracy'])

    
    filename = "_".join([cfg.model.model_name,
                         str(cfg.model.img_size),
                         "weights.{epoch:02d}-{val_loss:.2f}.hdf5"])
    callbacks.extend([
        LearningRateScheduler(schedule=scheduler),
        saveLossCallback(),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])
    if weight_file:
        model.load_weights(str(checkpoint_dir) + "/" + weight_file)
    model.fit(train_gen, epochs=cfg.train.epochs, callbacks=callbacks, validation_data=val_gen,
              workers=multiprocessing.cpu_count())


if __name__ == '__main__':
    main()
