import numpy as np
import os
import pickle
import torch
import time
import gymnasium.spaces as spaces
from torch import optim
from buffer import Buffer
from ppomodel import ActorCriticModel
from worker import Worker
from utils import create_env
from utils import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter


def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """
    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


class PPOTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.run_id = run_id
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"])
        self.observation_space = dummy_env.observation_space
        if isinstance(dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (dummy_env.action_space.n,)
            self.continuous = False
        if isinstance(dummy_env.action_space, spaces.Box):
            self.action_space_shape = dummy_env.action_space.shape
            self.continuous = True
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        if not self.continuous:
            self.buffer = Buffer(self.config, self.observation_space, self.action_space_shape, self.device, continuous=False)
        else:
            self.buffer = Buffer(self.config, self.observation_space, torch.zeros(np.prod(self.action_space_shape)), self.device, continuous=True)

        # Init model
        print("Step 3: Init model and optimizer")
        self.model = ActorCriticModel(self.config, self.observation_space, self.action_space_shape, self.continuous).to(self.device)
        self.model.train()
        if self.recurrence['layer_type'] == 's4':
            self.optimizer, self.scheduler = setup_optimizer(self.model, self.lr_schedule['initial'], weight_decay=0.01, epochs=400)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.config["n_workers"])]

        # Setup observation placeholder   
        self.obs = np.zeros((self.config["n_workers"],) + self.observation_space.shape, dtype=np.float32)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_cell = hxs
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_cell = (hxs, cxs)
        elif self.recurrence["layer_type"] == 's4':
            self.recurrent_cell = hxs

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        # Store episode results for monitoring statistics
        episode_infos = deque(maxlen=100)

        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # Train epochs
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)

            # Store recent episode infos
            episode_infos.extend(sampled_episode_info)
            episode_result = self._process_episode_info(episode_infos)

            # Print training statistics
            if "success_percent" in episode_result:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} success = {:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], episode_result["success_percent"],
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            else:
                result = "{:4} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} pi_loss={:3f} v_loss={:3f} entropy={:.3f} loss={:3f} value={:.3f} advantage={:.3f}".format(
                    update, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"], 
                    training_stats[0], training_stats[1], training_stats[3], training_stats[2], torch.mean(self.buffer.values), torch.mean(self.buffer.advantages))
            print(result)

            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result)
            
            # Free memory
            del(self.buffer.samples_flat)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Save the initial observations and recurrentl cell states
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                    self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                policy, value, self.recurrent_cell = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
                self.buffer.values[:, t] = value

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                if self.continuous:
                    self.buffer.actions[:, t, :] = actions[0]
                    self.buffer.log_probs[:, t, :] = log_probs[0]
                else:
                    self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                    self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Reset recurrent cell states
                    if self.recurrence["reset_hidden_state"]:
                        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                        if self.recurrence["layer_type"] == "gru":
                            self.recurrent_cell[:, w] = hxs
                        elif self.recurrence["layer_type"] == "lstm":
                            self.recurrent_cell[0][:, w] = hxs
                            self.recurrent_cell[1][:, w] = cxs
                # Store latest observations
                self.obs[w] = obs
                            
        # Calculate advantages
        _, last_value, _ = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
            if self.recurrence['layer_type'] == 's4':
                self.scheduler.step()
        return train_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        elif self.recurrence["layer_type"] == "s4":
            recurrent_cell = None

        # Forward model
        policy, value, _ = self.model(samples["obs"], recurrent_cell, self.device, self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        # print("action shape", samples["actions"].shape, "mask:", samples["loss_mask"].shape)
        for i, policy_branch in enumerate(policy):
            act = samples["actions"][:, i] if not self.continuous else samples["actions"]
            log_probs.append(policy_branch.log_prob(act))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1)
        if not self.continuous:
            entropies = entropies.sum(1).reshape(-1)
        
        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]] 

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        if self.recurrence['layer_type'] != 's4':
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("losses/entropy", training_stats[3], update)
        self.writer.add_scalar("training/sequence_length", self.buffer.true_sequence_length, update)
        self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)

    @staticmethod
    def _process_episode_info(episode_info:list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        try:
            self.writer.close()
        except:
            pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)