# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""util function to create a tf_agent."""

from typing import Any
from collections.abc import Callable

import abc
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.behavioral_cloning import behavioral_cloning_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

from compiler_opt.rl import constant_value_network
from compiler_opt.rl.distributed import agent as distributed_ppo_agent

# from tf_agents.policies import tf_policy
# import numpy as np
# import tensorflow_probability as tfp
# from tf_agents.trajectories import policy_step
# from sklearn.ensemble import RandomForestClassifier

from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step
import tensorflow_probability as tfp
import numpy as np





class AgentConfig(metaclass=abc.ABCMeta):
  """Agent creation and data processing hook-ups."""

  def __init__(self, *, time_step_spec: types.NestedTensorSpec,
               action_spec: types.NestedTensorSpec):
    self._time_step_spec = time_step_spec
    self._action_spec = action_spec

  @property
  def time_step_spec(self):
    return self._time_step_spec

  @property
  def action_spec(self):
    return self._action_spec

  @abc.abstractmethod
  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Specific agent configs must implement this."""
    raise NotImplementedError()

  def get_policy_info_parsing_dict(
      self) -> dict[str, tf.io.FixedLenSequenceFeature]:
    """Return the parsing dict for the policy info."""
    return {}

  # pylint: disable=unused-argument
  def process_parsed_sequence_and_get_policy_info(
      self, parsed_sequence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Function to process parsed_sequence and to return policy_info.

    Args:
      parsed_sequence: A dict from feature_name to feature_value parsed from TF
        SequenceExample.

    Returns:
      A nested policy_info for given agent.
    """
    return {}


@gin.configurable
def create_agent(agent_config: AgentConfig,
                 preprocessing_layer_creator: Callable[[types.TensorSpec],
                                                       tf.keras.layers.Layer],
                 policy_network: types.Network):
  """Gin configurable wrapper of AgentConfig.create_agent.
  Works around the fact that class members aren't gin-configurable."""
  preprocessing_layers = tf.nest.map_structure(
      preprocessing_layer_creator, agent_config.time_step_spec.observation)
  return agent_config.create_agent(preprocessing_layers, policy_network)


@gin.configurable(module='agents')
class BCAgentConfig(AgentConfig):
  """Behavioral Cloning agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a behavioral_cloning_agent."""

    network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='QNetwork')

    return behavioral_cloning_agent.BehavioralCloningAgent(
        self.time_step_spec,
        self.action_spec,
        cloning_network=network,
        num_outer_dims=2)


@gin.configurable(module='agents')
class DQNAgentConfig(AgentConfig):
  """DQN agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a dqn_agent."""
    network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='QNetwork')

    return dqn_agent.DqnAgent(
        self.time_step_spec, self.action_spec, q_network=network)


@gin.configurable(module='agents')
class PPOAgentConfig(AgentConfig):
  """PPO/Reinforce agent configuration."""

  def create_agent(self, preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a ppo_agent."""

    actor_network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        name='ActorDistributionNetwork')

    critic_network = constant_value_network.ConstantValueNetwork(
        self.time_step_spec.observation, name='ConstantValueNetwork')

    return ppo_agent.PPOAgent(
        self.time_step_spec,
        self.action_spec,
        actor_net=actor_network,
        value_net=critic_network)

  def get_policy_info_parsing_dict(
      self) -> dict[str, tf.io.FixedLenSequenceFeature]:
    if tensor_spec.is_discrete(self._action_spec):
      return {
          'CategoricalProjectionNetwork_logits':
              tf.io.FixedLenSequenceFeature(
                  shape=(self._action_spec.maximum - self._action_spec.minimum +
                         1),
                  dtype=tf.float32)
      }
    else:
      return {
          'NormalProjectionNetwork_scale':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32),
          'NormalProjectionNetwork_loc':
              tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.float32)
      }

  def process_parsed_sequence_and_get_policy_info(
      self, parsed_sequence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if tensor_spec.is_discrete(self._action_spec):
      policy_info = {
          'dist_params': {
              'logits': parsed_sequence['CategoricalProjectionNetwork_logits']
          }
      }
      del parsed_sequence['CategoricalProjectionNetwork_logits']
    else:
      policy_info = {
          'dist_params': {
              'scale': parsed_sequence['NormalProjectionNetwork_scale'],
              'loc': parsed_sequence['NormalProjectionNetwork_loc']
          }
      }
      del parsed_sequence['NormalProjectionNetwork_scale']
      del parsed_sequence['NormalProjectionNetwork_loc']
    return policy_info


@gin.configurable(module='agents')
class DistributedPPOAgentConfig(PPOAgentConfig):
  """Distributed PPO/Reinforce agent configuration."""

  def _create_agent_implt(self, preprocessing_layers: tf.keras.layers.Layer,
                          policy_network: types.Network) -> tf_agent.TFAgent:
    """Creates a ppo_distributed agent."""
    actor_network = policy_network(
        self.time_step_spec.observation,
        self.action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=tf.keras.layers.Concatenate(),
        name='ActorDistributionNetwork')

    critic_network = constant_value_network.ConstantValueNetwork(
        self.time_step_spec.observation, name='ConstantValueNetwork')

    return distributed_ppo_agent.MLGOPPOAgent(
        self.time_step_spec,
        self.action_spec,
        optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4, epsilon=1e-5),
        actor_net=actor_network,
        value_net=critic_network,
        value_pred_loss_coef=0.0,
        entropy_regularization=0.01,
        importance_ratio_clipping=0.2,
        discount_factor=1.0,
        gradient_clipping=1.0,
        debug_summaries=False,
        value_clipping=None,
        aggregate_losses_across_replicas=True,
        loss_scaling_factor=1.0)

# class RandomForestPolicy(tf_policy.TFPolicy):
#   """Policy that queries a scikit-learn RandomForestClassifier."""

#   def __init__(self,
#                time_step_spec: types.NestedTensorSpec,
#                action_spec: types.NestedTensorSpec,
#                preprocessing_layers,
#                rf_model,
#                name: str = 'RandomForestPolicy'):
#     if RandomForestClassifier is None:
#       raise ImportError(
#           'RandomForestPolicy requires scikit-learn to be installed.')
#     self._rf_model = rf_model
#     self._preprocessing_layers = preprocessing_layers
#     self._num_actions = int(
#         action_spec.maximum - action_spec.minimum + 1) if tensor_spec.is_discrete(
#             action_spec) else None

#     super().__init__(
#         time_step_spec=time_step_spec,
#         action_spec=action_spec,
#         policy_state_spec=(),
#         info_spec={},
#         name=name)

#   def _apply_preprocessing(self, observation):
#     """Apply preprocessing layers and flatten to [B, D]."""
#     processed = tf.nest.map_structure(
#         lambda layer, obs: layer(obs) if layer is not None else obs,
#         self._preprocessing_layers,
#         observation)

#     flat = tf.nest.flatten(processed)
#     flat = [tf.cast(f, tf.float32) for f in flat]

#     if len(flat) == 1:
#       x = flat[0]
#       if x.shape.rank == 1:
#         x = tf.expand_dims(x, axis=-1)
#       return x

#     pieces = []
#     for f in flat:
#       if f.shape.rank == 1:
#         f = tf.expand_dims(f, axis=-1)
#       else:
#         f = tf.reshape(f, [tf.shape(f)[0], -1])
#       pieces.append(f)

#     return tf.concat(pieces, axis=-1)

#   def _rf_predict(self, features):
#     """NumPy helper for RF prediction (used via tf.numpy_function)."""
#     preds = self._rf_model.predict(features)  # [B]
#     return preds.astype(np.int64)


#   def _rf_predict_proba(self, features):
#     probs = self._rf_model.predict_proba(features)  # [B, num_actions]
#     return probs.astype(np.float32)

#   def _action(self, time_step, policy_state, seed=None):
#     del seed
#     obs = time_step.observation
#     feats = self._apply_preprocessing(obs)

#     actions = tf.numpy_function(
#         func=self._rf_predict, inp=[feats], Tout=tf.int64)

#     batch_size = tf.shape(feats)[0]
#     if self._action_spec.shape.rank == 0:
#       actions = tf.reshape(actions, [batch_size])
#     else:
#       actions = tf.reshape(
#           actions,
#           tf.concat([[batch_size], self._action_spec.shape], axis=0))

#     return policy_step.PolicyStep(actions, policy_state, {})

#   def _distribution(self, time_step, policy_state):
#     obs = time_step.observation
#     feats = self._apply_preprocessing(obs)

#     probs = tf.numpy_function(
#         func=self._rf_predict_proba, inp=[feats], Tout=tf.float32)
#     batch_size = tf.shape(feats)[0]
#     probs = tf.reshape(probs, [batch_size, self._num_actions])

#     dist = tfp.distributions.Categorical(probs=probs)
#     return policy_step.PolicyStep(dist, policy_state, {})


# class RandomForestTFAgent(tf_agent.TFAgent):
#   """TFAgent that trains a RandomForestClassifier from experience."""

#   def __init__(self,
#                time_step_spec: types.NestedTensorSpec,
#                action_spec: types.NestedTensorSpec,
#                preprocessing_layers,
#                n_estimators: int = 100,
#                max_depth: int | None = None,
#                min_samples_leaf: int = 1,
#                retrain_every_samples: int = 10000,
#                name: str = 'RandomForestTFAgent'):
#     if RandomForestClassifier is None:
#       raise ImportError(
#           'RandomForestTFAgent requires scikit-learn to be installed.')

#     self._rf_model = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_leaf=min_samples_leaf,
#         random_state=0)

#     self._preprocessing_layers = preprocessing_layers
#     self._retrain_every = int(retrain_every_samples)
#     self._samples_since_retrain = 0

#     # Python-side buffers
#     self._buffer_X: list[np.ndarray] = []
#     self._buffer_y: list[np.ndarray] = []

#     # Dummy train_step_counter required by TFAgent
#     train_step_counter = tf.Variable(
#         0, dtype=tf.int64, trainable=False, name='train_step_counter')

#     rf_policy = RandomForestPolicy(
#         time_step_spec=time_step_spec,
#         action_spec=action_spec,
#         preprocessing_layers=preprocessing_layers,
#         rf_model=self._rf_model)

#     # Keep the name on the Python object, but DO NOT pass it to super().__init__
#     self._agent_name = name

#     super().__init__(
#         time_step_spec=time_step_spec,
#         action_spec=action_spec,
#         policy=rf_policy,
#         collect_policy=rf_policy,
#         train_sequence_length=None,
#         debug_summaries=False,
#         summarize_grads_and_vars=False,
#         train_step_counter=train_step_counter)


#   def _initialize(self):
#     # Nothing TF-trainable to initialize for RF.
#     return tf.no_op()

#   def _extract_xy(self, experience):
#     """Extract (X, y) from a Trajectory-like `experience`."""
#     obs = experience.observation
#     actions = experience.action

#     def flatten_bt(t):
#       # Flatten [B, T, ...] -> [B*T, ...]
#       if t.shape.rank is None or t.shape.rank < 2:
#         return t
#       shape = tf.shape(t)
#       bt = shape[0] * shape[1]
#       rest = shape[2:]
#       return tf.reshape(t, tf.concat([[bt], rest], axis=0))

#     obs_flat = tf.nest.map_structure(flatten_bt, obs)
#     actions_flat = flatten_bt(actions)

#     # Reuse same preprocessing as policy.
#     processed = tf.nest.map_structure(
#         lambda layer, o: layer(o) if layer is not None else o,
#         self._preprocessing_layers,
#         obs_flat)

#     flat_proc = tf.nest.flatten(processed)
#     flat_proc = [tf.cast(f, tf.float32) for f in flat_proc]

#     if len(flat_proc) == 1:
#       feats = flat_proc[0]
#       if feats.shape.rank == 1:
#         feats = tf.expand_dims(feats, axis=-1)
#     else:
#       pieces = []
#       for f in flat_proc:
#         if f.shape.rank == 1:
#           f = tf.expand_dims(f, axis=-1)
#         else:
#           f = tf.reshape(f, [tf.shape(f)[0], -1])
#         pieces.append(f)
#       feats = tf.concat(pieces, axis=-1)

#     # Convert to NumPy (assuming eager mode).
#     X_np = feats.numpy()
#     y_np = actions_flat.numpy()
#     return X_np, y_np

#   def _train(self, experience, weights=None):
#     del weights  # unused

#     X_batch, y_batch = self._extract_xy(experience)
#     self._buffer_X.append(X_batch)
#     self._buffer_y.append(y_batch)
#     self._samples_since_retrain += X_batch.shape[0]

#     # When enough new samples have accumulated, refit the RF.
#     if self._samples_since_retrain >= self._retrain_every:
#       X_all = np.concatenate(self._buffer_X, axis=0)
#       y_all = np.concatenate(self._buffer_y, axis=0)
#       self._rf_model.fit(X_all, y_all)
#       self._samples_since_retrain = 0

#     # Dummy scalar loss so Trainer is happy.
#     loss = tf.constant(0.0, dtype=tf.float32)
#     return tf_agent.LossInfo(loss=loss, extra=())


# @gin.configurable(module='agents')
# class RandomForestAgentConfig(AgentConfig):
#   """AgentConfig that builds and trains a RandomForest-based agent.

#   Uses dummy time_step_spec and action_spec if they are not provided.
#   """

#   def __init__(self,
#                *,
#                time_step_spec: types.NestedTensorSpec | None = None,
#                action_spec: types.NestedTensorSpec | None = None,
#                n_estimators: int = 100,
#                max_depth: int | None = None,
#                min_samples_leaf: int = 1,
#                retrain_every_samples: int = 10000):
#     # ----- define dummy specs if not supplied -----
#     if time_step_spec is None:
#       # Single float feature as a minimal dummy observation spec
#       time_step_spec = tf.TensorSpec(
#           shape=(1,), dtype=tf.float32, name='rf_observation')

#     if action_spec is None:
#       # Binary action: 0 or 1 (e.g., don't-inline / inline)
#       action_spec = tensor_spec.BoundedTensorSpec(
#           shape=(),
#           dtype=tf.int64,
#           minimum=0,
#           maximum=1,
#           name='rf_action')

#     # Base class stores them and exposes .time_step_spec / .action_spec
#     super().__init__(time_step_spec=time_step_spec, action_spec=action_spec)

#     # RF hyperparameters (gin-configurable)
#     self._n_estimators = n_estimators
#     self._max_depth = max_depth
#     self._min_samples_leaf = min_samples_leaf
#     self._retrain_every_samples = retrain_every_samples

#   def create_agent(self,
#                    preprocessing_layers: tf.keras.layers.Layer,
#                    policy_network: types.Network) -> tf_agent.TFAgent:
#     # policy_network is unused for RF
#     del policy_network
#     return RandomForestTFAgent(
#         time_step_spec=self.time_step_spec,
#         action_spec=self.action_spec,
#         preprocessing_layers=preprocessing_layers,
#         n_estimators=self._n_estimators,
#         max_depth=self._max_depth,
#         min_samples_leaf=self._min_samples_leaf,
#         retrain_every_samples=self._retrain_every_samples)

def _flatten_numeric_observation(observation):
  """Flatten all numeric tensors in observation into [B, D] float32."""
  flat_tensors = []
  for t in tf.nest.flatten(observation):
    if t.dtype.is_floating or t.dtype.is_integer:
      x = tf.cast(t, tf.float32)
      if x.shape.rank == 1:
        x = tf.expand_dims(x, axis=-1)
      else:
        x = tf.reshape(x, [tf.shape(x)[0], -1])
      flat_tensors.append(x)

  if not flat_tensors:
    raise ValueError('RandomForest: no numeric tensors in observation.')

  return tf.concat(flat_tensors, axis=-1)

class SoftDecisionTreeLayer(tf.keras.layers.Layer):
  """Differentiable decision tree of given depth for discrete actions.

  depth:
    number of decision levels; num_leaves = 2**depth.

  We represent a full binary tree in array form:
    node 0 is root,
    left child of i is 2*i + 1,
    right child of i is 2*i + 2.
  """

  def __init__(self, depth: int, num_actions: int, name: str = 'SoftDecisionTree'):
    super().__init__(name=name)
    if depth < 1:
      raise ValueError('depth must be >= 1')
    self._depth = depth
    self._num_actions = num_actions

    self._num_internal = 2 ** depth - 1
    self._num_leaves = 2 ** depth

    # One dense layer producing decision scores for all internal nodes.
    self._decision_layer = tf.keras.layers.Dense(
        units=self._num_internal,
        activation=None,
        use_bias=True,
        name=f'{name}_decision_dense')

    # Leaf logits: [num_leaves, num_actions].
    initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    self._leaf_logits = self.add_weight(
        name=f'{name}_leaf_logits',
        shape=(self._num_leaves, self._num_actions),
        initializer=initializer,
        trainable=True)

    # Precompute path structure: for each leaf, which nodes are on its path,
    # and whether we go right (1.0) or left (0.0) at each node.
    leaf_node_indices = np.zeros((self._num_leaves, self._depth), dtype=np.int32)
    leaf_is_right = np.zeros((self._num_leaves, self._depth), dtype=np.float32)

    for leaf in range(self._num_leaves):
      node = 0
      for d in range(self._depth):
        # Take bits from most-significant to least-significant:
        # 0 = left, 1 = right.
        bit = (leaf >> (self._depth - 1 - d)) & 1
        leaf_node_indices[leaf, d] = node
        leaf_is_right[leaf, d] = float(bit)
        if bit == 0:
          node = 2 * node + 1
        else:
          node = 2 * node + 2

    self._leaf_node_indices = tf.constant(leaf_node_indices, dtype=tf.int32)   # [L, D]
    self._leaf_is_right = tf.constant(leaf_is_right, dtype=tf.float32)         # [L, D]

  def call(self, features, training=False):
    """features: [B, D] -> logits: [B, num_actions]."""
    # Decision logits for all internal nodes.
    scores = self._decision_layer(features)          # [B, N_int]
    p_right = tf.sigmoid(scores)                     # [B, N_int]
    p_left = 1.0 - p_right                           # [B, N_int]

    # Expand: [B, 1, N_int] so we can gather by leaf/node indices.
    p_right_exp = p_right[:, tf.newaxis, :]          # [B, 1, N_int]
    p_left_exp = p_left[:, tf.newaxis, :]            # [B, 1, N_int]

    # Flatten leaf-node indices for a single gather.
    indices_flat = tf.reshape(self._leaf_node_indices, [-1])  # [L*D]

    # Gather right/left probabilities along node axis.
    right_flat = tf.gather(p_right_exp, indices_flat, axis=2)  # [B, 1, L*D]
    left_flat = tf.gather(p_left_exp, indices_flat, axis=2)    # [B, 1, L*D]

    # Reshape to [B, L, D].
    right = tf.reshape(right_flat, [tf.shape(features)[0], self._num_leaves, self._depth])
    left = tf.reshape(left_flat, [tf.shape(features)[0], self._num_leaves, self._depth])

    # Leaf direction mask: [1, L, D].
    leaf_is_right = self._leaf_is_right[tf.newaxis, ...]  # [1, L, D]

    # Probability of each decision along path.
    path_probs = leaf_is_right * right + (1.0 - leaf_is_right) * left  # [B, L, D]

    # Leaf probability is product over depth.
    leaf_prob = tf.reduce_prod(path_probs, axis=2)     # [B, L]

    # Turn into weights for leaf logits.
    leaf_prob_exp = leaf_prob[..., tf.newaxis]         # [B, L, 1]
    leaf_logits = self._leaf_logits[tf.newaxis, ...]   # [1, L, A]
    per_leaf_logits = leaf_prob_exp * leaf_logits      # [B, L, A]

    # Sum over leaves → logits for this tree: [B, A]
    logits = tf.reduce_sum(per_leaf_logits, axis=1)
    return logits

class RandomForestModel(tf.keras.Model):
  """Ensemble of soft decision trees with configurable depth."""

  def __init__(self,
               num_trees: int,
               depth: int,
               num_actions: int,
               name: str = 'RandomForestModel'):
    super().__init__(name=name)
    self._num_trees = num_trees
    self._depth = depth
    self._num_actions = num_actions

    self._trees = []
    for i in range(num_trees):
      tree = SoftDecisionTreeLayer(
          depth=depth,
          num_actions=num_actions,
          name=f'{name}_tree_{i}')
      self._trees.append(tree)

  def call(self, features, training=False):
    """features: [B, D] -> logits: [B, num_actions]."""
    logits_list = []
    for tree in self._trees:
      logits_list.append(tree(features, training=training))  # [B, A]

    stacked = tf.stack(logits_list, axis=1)  # [B, T, A]
    logits = tf.reduce_mean(stacked, axis=1) # [B, A]
    return logits

class RandomForestPolicy(tf_policy.TFPolicy):
  """TF-Agents Policy using a pure-TF RandomForestModel."""

  def __init__(self,
               time_step_spec: types.NestedTensorSpec,
               action_spec: types.NestedTensorSpec,
               rf_model: tf.keras.Model,
               name: str = 'RandomForestPolicy'):
    self._rf_model = rf_model
    self._num_actions = int(
        action_spec.maximum - action_spec.minimum + 1) if tensor_spec.is_discrete(
            action_spec) else None

    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=(),
        info_spec={},
        name=name)

  def _action(self, time_step, policy_state, seed=None):
    del seed
    obs = time_step.observation
    features = _flatten_numeric_observation(obs)          # [B, D]
    logits = self._rf_model(features, training=False)     # [B, A]

    actions = tf.argmax(
        logits, axis=-1, output_type=self._action_spec.dtype)  # [B]

    batch_size = tf.shape(features)[0]
    action_shape = self._action_spec.shape
    if len(action_shape) == 0:
      actions = tf.reshape(actions, [batch_size])
    else:
      actions = tf.reshape(
          actions, tf.concat([[batch_size], action_shape], axis=0))

    return policy_step.PolicyStep(actions, policy_state, {})

  def _distribution(self, time_step, policy_state):
    obs = time_step.observation
    features = _flatten_numeric_observation(obs)
    logits = self._rf_model(features, training=False)      # [B, A]
    dist = tfp.distributions.Categorical(logits=logits)
    return policy_step.PolicyStep(dist, policy_state, {})

class RandomForestTFAgent(tf_agent.TFAgent):
  """TFAgent that trains a RandomForestModel from experience."""

  def __init__(self,
               time_step_spec: types.NestedTensorSpec,
               action_spec: types.NestedTensorSpec,
               num_trees: int = 8,
               depth: int = 3,
               learning_rate: float = 1e-3,
               name: str = 'RandomForestTFAgent'):
    if not tensor_spec.is_discrete(action_spec):
      raise ValueError('RandomForestTFAgent only supports discrete actions.')

    self._num_actions = int(action_spec.maximum - action_spec.minimum + 1)

    # Dummy train_step_counter required by TFAgent.
    train_step_counter = tf.Variable(
        0, dtype=tf.int64, trainable=False, name='train_step_counter')

    # Forest model.
    self._rf_model = RandomForestModel(
        num_trees=num_trees,
        depth=depth,
        num_actions=self._num_actions,
        name='rf_model')

    # Optimizer for supervised training.
    self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    rf_policy = RandomForestPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        rf_model=self._rf_model)

    # Keep Python-side name, but DON'T pass it to TFAgent.__init__ (older TF-Agents).
    self._agent_name = name

    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy=rf_policy,
        collect_policy=rf_policy,
        train_sequence_length=None,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        train_step_counter=train_step_counter)

  def _initialize(self):
    return tf.no_op()

  def _flatten_experience(self, experience):
    """Flatten [B, T, ...] to [N, ...] and build (features, actions)."""
    obs = experience.observation
    actions = experience.action

    def flatten_bt(t):
      if t.shape.rank is None or t.shape.rank < 2:
        return t
      shape = tf.shape(t)
      bt = shape[0] * shape[1]
      rest = shape[2:]
      return tf.reshape(t, tf.concat([[bt], rest], axis=0))

    obs_flat = tf.nest.map_structure(flatten_bt, obs)
    actions_flat = flatten_bt(actions)  # [N]

    features = _flatten_numeric_observation(obs_flat)  # [N, D]
    return features, actions_flat

  def _train(self, experience, weights=None):
    del weights

    with tf.GradientTape() as tape:
      features, actions = self._flatten_experience(experience)
      logits = self._rf_model(features, training=True)  # [N, A]

      actions = tf.cast(actions, tf.int32)
      one_hot = tf.one_hot(actions, depth=self._num_actions)
      per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=one_hot, logits=logits)
      loss = tf.reduce_mean(per_example_loss)

    grads = tape.gradient(loss, self._rf_model.trainable_variables)
    self._optimizer.apply_gradients(
        zip(grads, self._rf_model.trainable_variables))

    self.train_step_counter.assign_add(1)

    return tf_agent.LossInfo(loss=loss, extra=())

@gin.configurable(module='agents')
class RandomForestAgentConfig(AgentConfig):
  """AgentConfig that builds and trains a RandomForest-based TFAgent."""

  def __init__(self,
               *,
               time_step_spec: types.NestedTensorSpec,
               action_spec: types.NestedTensorSpec,
               num_trees: int = 8,
               depth: int = 3,
               learning_rate: float = 1e-3):
    super().__init__(time_step_spec=time_step_spec, action_spec=action_spec)
    self._num_trees = num_trees
    self._depth = depth
    self._learning_rate = learning_rate

  def create_agent(self,
                   preprocessing_layers: tf.keras.layers.Layer,
                   policy_network: types.Network) -> tf_agent.TFAgent:
    # RF ignores preprocessing_layers and policy_network; it works on raw obs.
    del preprocessing_layers, policy_network
    return RandomForestTFAgent(
        time_step_spec=self.time_step_spec,
        action_spec=self.action_spec,
        num_trees=self._num_trees,
        depth=self._depth,
        learning_rate=self._learning_rate)

