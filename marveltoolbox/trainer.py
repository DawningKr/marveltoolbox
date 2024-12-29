import torch
import os
import logging
import traceback
from . import utils


class BaseTrainer:
    def __init__(self, confs):
        self.flag = confs.flag
        self.save_flag = confs.flag
        self.checkpoint_path = confs.checkpoint_path
        self.log_path = confs.log_path
        self.device = confs.device
        self.batch_size = confs.batch_size
        self.epochs = confs.epochs
        self.seed = confs.seed
        self.device_ids = confs.device_ids
        self.start_epoch = 0
        self.schedulers = {}
        self.models = {}
        self.optimizers = {}
        self.datasets = {}
        self.train_sets = {}
        self.eval_sets = {}
        self.dataloaders = {}
        self.records = {}
        self.logs = {}
        self.model_names = self.models.keys()
        self.logger = None

        if not os.path.exists(self.checkpoint_path):
            print(f"{self.checkpoint_path} dose not exist. Creating...")
            os.makedirs(self.checkpoint_path)

        if not os.path.exists(self.log_path):
            print(f"{self.log_path} dose not exist. Creating...")
            os.makedirs(self.log_path)

    def set_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_file_name = "{}.log".format(self.save_flag)
        log_file_path = os.path.join(self.log_path, log_file_name)
        print(f"Log file save at: {log_file_path}")
        handler_file = logging.FileHandler(log_file_path)
        self.logger.addHandler(handler_file)

    def preprocessing(self):
        kwargs = (
            {"num_workers": 1, "drop_last": True, "pin_memory": True}
            if torch.cuda.is_available()
            else {}
        )
        for key in self.datasets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.datasets[key], batch_size=self.batch_size, shuffle=True, **kwargs
            )

        for key in self.train_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.train_sets[key], batch_size=self.batch_size, shuffle=True, **kwargs
            )

        kwargs = (
            {"num_workers": 1, "drop_last": False, "pin_memory": True}
            if torch.cuda.is_available()
            else {}
        )
        for key in self.eval_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.eval_sets[key], batch_size=self.batch_size, shuffle=True, **kwargs
            )

    def train(self, epoch):
        return 0.0

    def evaluate(self, epoch):
        return False

    def scheduler_step(self):
        pass

    def print_logs(self, epoch, step):
        msg = "Epoch/Iteration:{:0>3d}/{:0>4d} ".format(epoch, step)
        for key, value in self.logs.items():
            if type(value) == str:
                msg += "{}:{} ".format(key, value)
            else:
                msg += "{}:{:4f} ".format(key, value)
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def main(self, load_best=False, retrain=False, del_logger=True):
        utils.set_seed(self.seed)
        self.set_logger()
        if not retrain:
            self.load()
        timer = utils.Timer(self.epochs - self.start_epoch, self.logger)
        timer.init()
        for epoch in range(self.start_epoch, self.epochs):
            loss = self.train(epoch)
            is_best = self.evaluate(epoch)
            timer.step()
            self.save(is_best=is_best)
            self.start_epoch += 1
            self.scheduler_step()

        if load_best:
            self.load(is_best=True)
            msg = "Best epoch: {:0>3d} \n".format(self.start_epoch)
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)

        if del_logger:
            self.logger = None

    def run(self, *args, **kwargs):
        try:
            self.main(*args, **kwargs)
        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)

    def save(self, is_best=False):
        state_dict = {}
        state_dict["epoch"] = self.start_epoch + 1
        state_dict["records"] = self.records

        for name in self.optimizers.keys():
            state_dict["optim_{}".format(name)] = self.optimizers[name].state_dict()

        for name in self.models.keys():
            state_dict["model_{}".format(name)] = self.models[name].state_dict()

        for name in self.schedulers.keys():
            state_dict["scheduler_{}".format(name)] = self.schedulers[name].state_dict()

        utils.save_checkpoint(
            state_dict, is_best, file_path=self.checkpoint_path, flag=self.save_flag
        )

    def load(self, is_best=False):
        checkpoint = utils.load_checkpoint(
            is_best, file_path=self.checkpoint_path, flag=self.save_flag
        )
        if checkpoint:
            self.start_epoch = checkpoint["epoch"]
            self.records = checkpoint["records"]

            for name in self.optimizers.keys():
                self.optimizers[name].load_state_dict(
                    checkpoint["optim_{}".format(name)]
                )

            for name in self.models.keys():
                self.models[name].load_state_dict(
                    checkpoint["model_{}".format(name)], strict=False
                )

            for name in self.schedulers.keys():
                self.schedulers[name].load_state_dict(
                    checkpoint["scheduler_{}".format(name)]
                )
            msg = "=> loaded checkpoint (epoch {})".format(checkpoint["epoch"])
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)
