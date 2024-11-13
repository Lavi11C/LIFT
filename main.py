import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # output file setting
    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    # seed setting -> 確保結果一致
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    
    # 檢查是某指定了一個已訓練的模型路徑 -> 搭配以下使用
    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir)
    
    # 評估模型的零樣本性能
    if cfg.zero_shot:
        trainer.test()
        return

    # 在訓練集上測試模型(判斷是否過度擬合)
    if cfg.test_train == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_train_True")]
            print("Model directory: {}".format(cfg.model_dir))

        trainer.load_model(cfg.model_dir)
        trainer.test("train")
        return

    # 在測試集上測試模型(判斷泛化能力)
    if cfg.test_only == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_only_True")]
            print("Model directory: {}".format(cfg.model_dir))
        
        trainer.load_model(cfg.model_dir)
        trainer.test()
        return

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
    