import argparse, os, pytz
from datetime import datetime as dt
from yacs.config import CfgNode
from default_config import (
    node2dict, get_default_config, datamanage_args, engine_args
)
from postprocess import compose_best_model, summarize_data_structure
from modules.data import ImageDataManager, Market1501
from modules.models import build_model
from modules.optim import build_optimizer
from modules.engine import Engine
from modules.losses import SoftmaxLoss, TripletLoss
from modules.utils import setup_logger, Logger, WandbSummary, dir2zip
from interface.driver import S3BucketDriver


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--root', type=str, required=True, help='dataset path for training'
    )

    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )

    parser.add_argument(
        '--exp-name', type=str, default='', help='experiment name'
    )

    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )

    # parse arguments & update configs
    args = parser.parse_args()
    cfg = get_default_config()

    # update configs with config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)


    # add `use_gpu` key to specified items
    for item in [cfg.data, cfg.model, cfg.loss.softmax, cfg.loss.triplet]:
        item.update({'use_gpu': cfg.setup.use_gpu})

    # start logging
    JST = dt.now(pytz.timezone('Asia/Tokyo'))
    experiment_name = args.exp_name + JST.strftime('@train-%Y-%m-%d-%H-%M-%S')
    logf = os.path.join(cfg.train.save_dir, 'results', experiment_name + '.txt')
    logger = setup_logger(logf)

    # logging with wandb
    if cfg.setup.log_wandb:
        writer = WandbSummary(exp_name=experiment_name, **node2dict(cfg))
    else:
        writer = None

    # print params
    print('\n--- Show configuration ---\n{}\n'.format(cfg))
    print('-> Target model: {}\n'.format(cfg.model.name))
    print('-> Build loss: {}\n'.format(cfg.loss.name))

    train_exec(cfg, args, writer, logger) # CALL Train



def build_lossfunc(n_classes:int, embed_size:int, cfg:CfgNode):
    """ building loss function """

    if cfg.loss.name == 'softmax':
        loss_cfg = node2dict(cfg.loss.softmax)
        return SoftmaxLoss(n_classes, embed_size, **loss_cfg)
    else:
        loss_cfg = node2dict(cfg.loss.triplet)
        return TripletLoss(n_classes, embed_size, **loss_cfg)



def train_exec(
    cfg:CfgNode, 
    args:argparse.ArgumentParser, 
    writer:WandbSummary, 
    logger:Logger
):
    """ model trainer """

    device = 'cuda' if cfg.setup.use_gpu else 'cpu'

    dataholder = Market1501(root=args.root)

    datamanager = ImageDataManager(dataholder, **datamanage_args(cfg))
    print('n_classes: {}\n'.format(datamanager.num_train_pids))

    model = build_model(**node2dict(cfg.model)).to(device)

    lossfunc = build_lossfunc(datamanager.num_train_pids, model.feature_dim, cfg)

    optimizer = build_optimizer(model, lossfunc, **node2dict(cfg.optimizer))

    engine = Engine(
        datamanager = datamanager, 
        model = model, 
        optimizer = optimizer, 
        lossfunc = lossfunc,
        use_gpu = cfg.setup.use_gpu,
        writer = writer,
    )
    engine.run(**engine_args(cfg)) # ***

    compose_best_model(cfg=cfg)
    summarize_data_structure(cfg=cfg, datamanager=datamanager)
    logger.flush()

    if writer:
        # experiments name
        exp_name = f'{engine.writer.exp_name}@{engine.writer.run.id}'

        # transfer results data
        results_dir = os.path.join(cfg.train.save_dir, 'results')
        dir2zip(results_dir)
        S3BucketDriver('results').post(results_dir + '.zip', rename=exp_name)

        # finish wandb logging
        engine.writer.alert('Completed training model: {}'.format(exp_name))
        engine.writer.finish()



if __name__ == '__main__':
    main()