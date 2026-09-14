"""Asynchronous environment wrapper for prefetching during training.

Implements double-buffered async sampling to overlap environment stepping
with GPU computation.
"""

from typing import Tuple, Dict, Any, Optional, List, Callable
from multiprocessing import Process, Queue, Event
from queue import Empty
import numpy as np
import threading
import time

from envs.maze_env import MazeEnv


class AsyncEnvWorker:
    """Worker that runs environments asynchronously and prefetches transitions."""

    def __init__(
        self,
        env_fn: Callable[[], MazeEnv],
        action_queue: Queue,
        result_queue: Queue,
        stop_event: Event,
        worker_id: int,
    ):
        """Initialize the async worker.

        Args:
            env_fn: Factory function to create environment.
            action_queue: Queue to receive actions from.
            result_queue: Queue to send results to.
            stop_event: Event to signal worker shutdown.
            worker_id: Unique identifier for this worker.
        """
        self.env_fn = env_fn
        self.action_queue = action_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.worker_id = worker_id

    def run(self) -> None:
        """Main worker loop."""
        env = self.env_fn()
        obs, info = env.reset()

        # Send initial observation
        self.result_queue.put({
            'type': 'reset',
            'worker_id': self.worker_id,
            'obs': obs,
            'info': info,
        })

        while not self.stop_event.is_set():
            try:
                # Wait for action with timeout to check stop_event
                action = self.action_queue.get(timeout=0.1)

                if action is None:  # Shutdown signal
                    break

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Auto-reset on episode end
                if terminated or truncated:
                    final_obs = next_obs
                    next_obs, reset_info = env.reset()
                    info['final_observation'] = final_obs
                    info['_episode_done'] = True
                else:
                    info['_episode_done'] = False

                # Send result
                self.result_queue.put({
                    'type': 'step',
                    'worker_id': self.worker_id,
                    'obs': next_obs,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info,
                })

            except Empty:
                continue

        env.close()


def _worker_process(
    env_fn: Callable[[], MazeEnv],
    action_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    worker_id: int,
) -> None:
    """Process target for async worker."""
    worker = AsyncEnvWorker(env_fn, action_queue, result_queue, stop_event, worker_id)
    worker.run()


class AsyncVecEnv:
    """Asynchronous vectorized environment with prefetching.

    Uses separate processes for each environment to enable true parallel
    environment stepping. Implements double-buffering to overlap env stepping
    with GPU training.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], MazeEnv]],
        prefetch_batches: int = 2,
    ):
        """Initialize the async vectorized environment.

        Args:
            env_fns: List of factory functions that create environments.
            prefetch_batches: Number of batches to prefetch.
        """
        self.num_envs = len(env_fns)
        self.prefetch_batches = prefetch_batches

        # Create queues for communication
        self.action_queues = [Queue() for _ in range(self.num_envs)]
        self.result_queue = Queue()
        self.stop_event = Event()

        # Start worker processes
        self.workers = []
        for i, env_fn in enumerate(env_fns):
            p = Process(
                target=_worker_process,
                args=(env_fn, self.action_queues[i], self.result_queue, self.stop_event, i),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        # Get initial observations
        self.observations = [None] * self.num_envs
        self._collect_results(count=self.num_envs, result_type='reset')

        # Get spaces from first env
        temp_env = env_fns[0]()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()

        # Pending results buffer
        self.pending_results = []

    def _collect_results(
        self,
        count: int,
        result_type: Optional[str] = None,
        timeout: float = 30.0,
    ) -> List[Dict]:
        """Collect results from workers.

        Args:
            count: Number of results to collect.
            result_type: Expected result type (None = any).
            timeout: Maximum wait time per result.

        Returns:
            List of result dictionaries.
        """
        results = []
        deadline = time.time() + timeout * count

        while len(results) < count:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for {count} results")

            try:
                result = self.result_queue.get(timeout=min(remaining, 1.0))
                if result_type is None or result['type'] == result_type:
                    results.append(result)
                    # Update stored observation
                    self.observations[result['worker_id']] = result['obs']
            except Empty:
                continue

        return results

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments.

        Note: This is a simplified reset - workers auto-reset on episode end.
        For explicit reset, recreate the AsyncVecEnv.

        Args:
            seed: Not used (workers handle their own seeding).

        Returns:
            Tuple of (stacked observations, list of info dicts).
        """
        observations = np.stack(self.observations)
        infos = [{'worker_id': i} for i in range(self.num_envs)]
        return observations, infos

    def step_async(self, actions: np.ndarray) -> None:
        """Send actions to all workers without waiting for results.

        Args:
            actions: Array of actions, one per environment.
        """
        for i, action in enumerate(actions):
            self.action_queues[i].put(int(action))

    def step_wait(self, timeout: float = 30.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Wait for all workers to complete their steps.

        Args:
            timeout: Maximum wait time.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos).
        """
        results = self._collect_results(self.num_envs, result_type='step', timeout=timeout)

        # Sort by worker_id to maintain consistent ordering
        results.sort(key=lambda x: x['worker_id'])

        observations = np.stack([r['obs'] for r in results])
        rewards = np.array([r['reward'] for r in results], dtype=np.float32)
        terminateds = np.array([r['terminated'] for r in results], dtype=np.bool_)
        truncateds = np.array([r['truncated'] for r in results], dtype=np.bool_)
        infos = [r['info'] for r in results]

        return observations, rewards, terminateds, truncateds, infos

    def step(
        self,
        actions: np.ndarray,
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
        """Close all environments and terminate workers."""
        self.stop_event.set()

        # Send shutdown signal to all workers
        for q in self.action_queues:
            try:
                q.put(None)
            except:
                pass

        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()


class PrefetchWrapper:
    """Wrapper that prefetches environment steps during training.

    Implements double-buffering to overlap environment stepping with
    GPU computation for maximum throughput.
    """

    def __init__(self, env: AsyncVecEnv, agent_select_fn: Callable):
        """Initialize the prefetch wrapper.

        Args:
            env: AsyncVecEnv to wrap.
            agent_select_fn: Function to select actions given observations.
        """
        self.env = env
        self.agent_select_fn = agent_select_fn

        self.current_obs = None
        self.pending_step = False

        # Thread for async prefetching
        self.prefetch_thread = None
        self.prefetch_result = None
        self.prefetch_lock = threading.Lock()

    def _prefetch_worker(self, actions: np.ndarray) -> None:
        """Background thread that executes environment step."""
        result = self.env.step(actions)
        with self.prefetch_lock:
            self.prefetch_result = result

    def start_prefetch(self, actions: np.ndarray) -> None:
        """Start prefetching the next step in background.

        Args:
            actions: Actions to execute.
        """
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()

        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(actions,)
        )
        self.prefetch_thread.start()

    def get_prefetch_result(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Get the result of the prefetched step (blocks if not ready).

        Returns:
            Step result tuple.
        """
        if self.prefetch_thread is not None:
            self.prefetch_thread.join()

        with self.prefetch_lock:
            result = self.prefetch_result
            self.prefetch_result = None

        return result

    def close(self) -> None:
        """Clean up resources."""
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
        self.env.close()


def make_async_vec_env(
    num_envs: int,
    rows: int = 20,
    cols: int = 20,
    stochasticity: float = 0.02,
    goal_reward: float = 200.0,
    oil_reward: float = -5.0,
    bump_reward: float = -10.0,
    action_reward: float = -1.0,
    seed: Optional[int] = None,
) -> AsyncVecEnv:
    """Create an async vectorized maze environment.

    Args:
        num_envs: Number of parallel environments.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        stochasticity: Probability of random action.
        goal_reward: Reward for reaching the goal.
        oil_reward: Penalty for stepping on oil.
        bump_reward: Penalty for stepping on bumps.
        action_reward: Cost per action taken.
        seed: Base seed for environments.

    Returns:
        Async vectorized environment.
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
    return AsyncVecEnv(env_fns)
