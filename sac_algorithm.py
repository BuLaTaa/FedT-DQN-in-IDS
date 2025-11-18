from torch import nn as nn
import torch
import rlkit.torch.pytorch_util as ptu
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import abc
import tqdm


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            start_epoch=0,  # negative epochs are offline, positive epochs are online
            name='default'
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            name
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.name = name
    
    def fuse(self, other):
        self.trainer.fuse(other.trainer)
    
    def get_networks(self):
        return self.trainer.networks[1:]
    
    def get_stats(self):
        return self.trainer.get_stats()
    
    def set_networks(self, networks):
        self.trainer.set_networks(networks)
    
    def to(self, device):
        self.trainer.to(device)
       
    def step(self, epoch):
        """Negative epochs are offline, positive epochs are online"""
        # for self.epoch in gt.timed_for(
        #         range(self._start_epoch, self.num_epochs),
        #         save_itrs=True,
        # ):

        offline_rl = epoch < 0
        self.to(ptu.device)

        self._begin_epoch(epoch)
        self._step(epoch, offline_rl)
        self._end_epoch(epoch)
        #self.trainer.to('cpu')

    def _step(self, epoch, offline_rl):
        if epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp(f'client_{self.client_id}_epoch_{epoch}_evaluation_sampling', unique=True)

        for _ in tqdm.tqdm(range(self.num_train_loops_per_epoch), desc='num_train_loops'):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            gt.stamp(f'client_{self.client_id}_epoch_{epoch}_exploration_sampling', unique=True)

            if not offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp(f'client_{self.client_id}_epoch_{epoch}_data_storing', unique=True)

            self.training_mode(True)
            gt.stamp(f'client_{self.client_id}_epoch_{epoch}_training_start', unique=True)  # 新增
            # 调试代码
            print(f"\n🔍 [客户端 {self.client_id}] 开始训练循环调试")
            print(f"Replay Buffer 大小: {self.replay_buffer.num_steps_can_sample()}")

            # 测试采样一个小批次
            try:
                print("🧪 测试采样小批次...")
                test_batch = self.replay_buffer.random_batch(2)  # 只采样2个样本
                print("✅ 小批次采样成功:")
                for key, value in test_batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: Tensor, shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    elif isinstance(value, np.ndarray):
                        print(f"  {key}: ndarray, shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            except Exception as e:
                print(f"❌ 小批次采样失败: {str(e)}")
                import traceback
                print(f"错误详情:\n{traceback.format_exc()}")
                raise
            for itr in tqdm.tqdm(range(self.num_trains_per_train_loop), desc='trains per train loop'):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)

            gt.stamp(f'client_{self.client_id}_epoch_{epoch}_training_end', unique=True)  # 新增
            self.training_mode(False)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def __init__(self, client_id, **kwargs):
        super().__init__(**kwargs)
        self.client_id = client_id  # 存储客户端ID
        self.name = f'client_{self.client_id}'
    def _end_epoch(self, epoch):
        """覆盖父类方法，避免生成默认名称"""
        snapshot = super()._get_snapshot()
        self.logger.save_itr_params(epoch, snapshot)
        gt.stamp(f'client_{self.client_id}_epoch_{epoch}_saving', unique=True)
        self._log_stats(epoch)
    def _log_stats(self, epoch):
        """完全覆盖父类方法，避免调用父类中的 gt.stamp"""
        # 1. 记录日志统计信息（保留必要逻辑）
        self.logger.record_dict({"epoch": epoch}, step=epoch)
        self.logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )
        self.logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/', step=epoch)
        # ...（其他日志记录逻辑，按需复制父类代码）...

        # 2. 自定义时间戳名称（包含客户端 ID 和轮次）
        gt.stamp(f'client_{self.client_id}_epoch_{epoch}_logging', unique=True)

        # 3. 禁用父类中的默认时间戳调用
        # （不再调用 super()._log_stats(epoch)）

        # 4. 输出日志
        self.logger.dump_tabular(with_prefix=False, with_timestamp=False)
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)