"""Vectorized environment wrappers for parallel environment execution."""

from typing import Tuple, Dict, Any, Optional, List, Callable
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy as np

from .maze_env import MazeEnv


def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn: Callable[[], MazeEnv]
) -> None:
    """Worker process that runs an environment.

    Args:
        remote: Connection to receive commands from.
        parent_remote: Parent connection to close.
        env_fn: Factory function to create the environment.
    """
    parent_remote.close()
    env = env_fn()

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                # Auto-reset on episode end
                if terminated or truncated:
                    final_obs = obs
                    final_info = info.copy()
                    obs, reset_info = env.reset()
                    info = {
                        'final_observation': final_obs,
                        'final_info': final_info,
                        '_terminal_observation': final_obs,
                    }
                    info.update(reset_info)
                remote.send((obs, reward, terminated, truncated, info))
            elif cmd == "reset":
                obs, info = env.reset(seed=data.get('seed') if data else None)
                remote.send((obs, info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "render":
                remote.send(env.render())
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
        except EOFError:
            break


class VecEnv:
    """Base class for vectorized environments."""

    def __init__(self, num_envs: int, observation_space, action_space):
        """Initialize the vectorized environment.

        Args:
            num_envs: Number of parallel environments.
            observation_space: Observation space of single env.
            action_space: Action space of single env.
        """
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments."""
        raise NotImplementedError

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        raise NotImplementedError

    def close(self) -> None:
        """Close all environments."""
        raise NotImplementedError


class SyncVecEnv(VecEnv):
    """Synchronous vectorized environment running envs sequentially.

    Useful for debugging and when the environment is very fast.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], MazeEnv]],
    ):
        """Initialize the synchronous vectorized environment.

        Args:
            env_fns: List of factory functions that create environments.
        """
        self.envs = [fn() for fn in env_fns]
        num_envs = len(self.envs)
        observation_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space
        super().__init__(num_envs, observation_space, action_space)

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments.

        Args:
            seed: Base seed for resetting (each env gets seed + i).

        Returns:
            Tuple of (stacked observations, list of info dicts).
        """
        observations = []
        infos = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments.

        Args:
            actions: Array of actions, one per environment.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos).
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            # Auto-reset on episode end
            if terminated or truncated:
                info['final_observation'] = obs
                info['_terminal_observation'] = obs
                obs, reset_info = env.reset()
                info.update(reset_info)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


class SubprocVecEnv(VecEnv):
    """Vectorized environment using subprocesses for true parallelism.

    Each environment runs in its own process, enabling parallel stepping
    for compute-heavy environments.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], MazeEnv]],
    ):
        """Initialize the subprocess vectorized environment.

        Args:
            env_fns: List of factory functions that create environments.
        """
        num_envs = len(env_fns)

        # Create pipes for parent-child communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        # Start worker processes
        self.processes = []
        for remote, work_remote, env_fn in zip(self.remotes, self.work_remotes, env_fns):
            process = Process(
                target=_worker,
                args=(work_remote, remote, env_fn),
                daemon=True
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        # Get spaces from first environment
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(num_envs, observation_space, action_space)

        self.waiting = False

    def reset(
        self,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments.

        Args:
            seed: Base seed for resetting (each env gets seed + i).

        Returns:
            Tuple of (stacked observations, list of info dicts).
        """
        for i, remote in enumerate(self.remotes):
            env_seed = {'seed': seed + i} if seed is not None else None
            remote.send(("reset", env_seed))

        results = [remote.recv() for remote in self.remotes]
        observations, infos = zip(*results)

        return np.stack(observations), list(infos)

    def step_async(self, actions: np.ndarray) -> None:
        """Send step commands to all environments without waiting.

        Args:
            actions: Array of actions, one per environment.
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Wait for all environments to complete their steps.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos).
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        observations, rewards, terminateds, truncateds, infos = zip(*results)

        return (
            np.stack(observations),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            list(infos),
        )

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments synchronously.

        Args:
            actions: Array of actions, one per environment.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos).
        """
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        """Close all environments and terminate worker processes."""
        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()


def make_vec_env(
    num_envs: int,
    rows: int = 20,
    cols: int = 20,
    stochasticity: float = 0.02,
    goal_reward: float = 200.0,
    oil_reward: float = -5.0,
    bump_reward: float = -10.0,
    action_reward: float = -1.0,
    use_subproc: bool = True,
    seed: Optional[int] = None,
) -> VecEnv:
    """Create a vectorized maze environment.

    Args:
        num_envs: Number of parallel environments.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        stochasticity: Probability of random action.
        goal_reward: Reward for reaching the goal.
        oil_reward: Penalty for stepping on oil.
        bump_reward: Penalty for stepping on bumps.
        action_reward: Cost per action taken.
        use_subproc: If True, use SubprocVecEnv for true parallelism.
        seed: Base seed for environments.

    Returns:
        Vectorized environment.
    """
    def make_env(env_idx: int) -> Callable[[], MazeEnv]:
        def _init() -> MazeEnv:
            env = MazeEnv(
                rows=rows,
                cols=cols,
                stochasticity=stochasticity,
                goal_reward=goal_reward,
                oil_reward=oil_reward,
                bump_reward=bump_reward,
                action_reward=action_reward,
            )
            if seed is not None:
                env.reset(seed=seed + env_idx)
            return env
        return _init

    env_fns = [make_env(i) for i in range(num_envs)]

    if use_subproc and num_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return SyncVecEnv(env_fns)
