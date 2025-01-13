import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import pandas as pd

from torchvision.utils import save_image
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch import loggers as pl_loggers

from taming.data.utils import custom_collate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="gpu idx",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="use debug mode",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class Test_Callback(Callback):
    def __init__(self, base_path, name, excel_folder, img_folder):
        super().__init__()
        self.dir = base_path
        self.pd_data = pd.DataFrame()
        self.excel_dir = os.path.join(base_path, excel_folder)
        self.img_dir = os.path.join(base_path, img_folder)
        self.name = name

    @rank_zero_only
    def log_information(self, output_dict, img_name, batch_idx):
        output_dict['name'] = img_name
        this_data = pd.DataFrame(output_dict, index=[batch_idx])
        self.pd_data = pd.concat([self.pd_data, this_data])
        
    @rank_zero_only
    def log_img(self, img_dict, img_name, batch_idx):
        img_name = img_name.split('.')[0]
        for k, v in img_dict.items():
            n = f"{batch_idx}_{img_name}_{k}.png"
            n = os.path.join(self.img_dir, n)
            save_image(v, n)

    def on_test_start(self, trainer, pl_module) -> None:
        if not os.path.exists(self.excel_dir):
            os.mkdir(self.excel_dir)
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        img_name = batch['file_path_'][0].split('/')[-1]
        if outputs['result']:
            self.log_information(outputs['result'], img_name, batch_idx)
        if outputs['imgs']:
            self.log_img(outputs['imgs'], img_name, batch_idx)
        
    def on_test_end(self, trainer, pl_module) -> None:
        detail_excel_name = os.path.join(self.excel_dir, f"details_{self.name}.xlsx")
        self.pd_data.to_excel(detail_excel_name)
        pd_avg = self.pd_data.mean(numeric_only=True).T
        avg_excel_name = os.path.join(self.excel_dir, f"average_{self.name}.xlsx")
        pd_avg.to_excel(avg_excel_name)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    
    # generate save path and file name
    if hasattr(config.model.params, "test_config"):
        preserve_type = config.model.params.test_config.mask_model.params.mask_type
        down_mask = config.model.params.test_config.mask_model.params.down_mask
        down_mask = "down_mask" if down_mask else "fullsize_mask"
        name = f"{preserve_type}_{down_mask}"
    else:
        name = "test"
    
    if opt.name:
        name = f"{opt.name}_{name}"
    if opt.debug:
        name = f"DEBUG_{name}"
    namenow = f"{name}_{now}"
    logdir = os.path.join("results", "tf_result", namenow)
    
    # set gpus
    if opt.gpus:
        print(f"Running on GPUs {opt.gpus}")
        trainer_config["distributed_backend"] = "ddp"
        trainer_config["gpus"] = opt.gpus
        ngpu = len(opt.gpus.strip(",").split(','))
        assert ngpu == 1        # only use one gpu for test!
    else:
        ngpu = 1
    
    # load model
    model = instantiate_from_config(config.model)
    
    # define trainer and callbacks dict
    trainer_kwargs = dict()

    # define logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir)

    # define trainer
    rd_callback = Test_Callback(logdir, name=namenow, excel_folder="excel", img_folder="image")
    trainer = Trainer(logger=tb_logger, callbacks=rd_callback,
                      devices=opt.gpus if opt.gpus else 'auto')
    
    # data
    config.data.params.batch_size = 1
    config.data.params.num_workers = 1
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    
    trainer.test(model, data)
    