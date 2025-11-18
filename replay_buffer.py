# replay_buffer.py
import abc
import numpy as np

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        # 获取labels（如果存在）
        labels = path.get("labels", None)
        
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path.get("agent_infos", [{}] * len(path["observations"])),
            path.get("env_infos", [{}] * len(path["observations"])),
        )):
            # 准备kwargs
            kwargs = {}
            
            # 添加agent_info和env_info
            kwargs.update(agent_info)
            kwargs.update(env_info)
            
            # 优先使用path中的labels数组
            if labels is not None and i < len(labels):
                kwargs['labels'] = labels[i]
            # 否则尝试从env_info中获取
            elif 'labels' in env_info:
                kwargs['labels'] = env_info['labels']
            elif 'true_label' in env_info:
                kwargs['labels'] = env_info['true_label']
            
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                **kwargs
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return