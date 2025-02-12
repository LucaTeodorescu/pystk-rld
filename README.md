# PySTK2-gymnasium / BBRL project template

This project template contains a basic structure that could be used for your PySTK2/BBRL project.
For information about the PySTK2 gymnasium environment, please look at the [corresponding github page](https://github.com/bpiwowar/pystk2-gymnasium)

## Structure

**Warning**: all the imports should be relative within your module (see `learn.py` for an example).

### `actors.py`

Contains the actors used throughout your project

### `learn.py`

Contains the code to train your actor

### `pystk_actor.py`

This Python file (**don't change its name**) should contain:

- `env_name`: The base environment name
- `player_name`: The actor name (displayed on top of the kart)
- `get_actor(state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space)`. It should return an actor that writes into `action` or `action/...`. It should *not* return a temporal agent. The parameters of the agent should be saved with `torch.save(actor.state_dict(), "pystk_actor.pth")`



### Learn your model

```sh
# To be run from the base directory
PYTHONPATH=. python -m stk_actor.learn
```

This should create the `pystk_actor.pth` file (**don't change its name**) that contains the parameters of your model. The file will be loaded using `torch.load(...)` and the data will be transmitted as  a parameter to `get_actor` (see `pystk_actor.py`).



# Testing the actor

## 1. Create the ZIP file

After learning your model, create the ZIP file containing the code and the actor parameters with:

```sh
# To be run from the base directory
(cd stk_actor; zip -r ../actor.zip .)
```

## 2. Test

You can use [master-dac](https://pypi.org/project/master_dac/) to test the zip file (you can even experiment with races between different actors to select the one of your choice):

```sh

# Usage: master-dac rld stk-race [OPTIONS] [ZIP_FILES|MODULE]...

master-dac rld stk-race --hide stk_actor.zip
```

To test your agent directly (for debug purposes), you can also use
```sh
PYTHONPATH=. master-dac rld stk-race --hide stk_actor
```

# Pystk2-Gymnasium Environment description

Main environment is `supertuxkart/full-v0` containing all the observations.

## `action_space`:

- `acceleration`: `Box(0.0, 1.0, (1,), float32)`
- `brake`: `Discrete(2)`
- `drift`: `Discrete(2)`
- `fire`: `Discrete(2)`
- `nitro`: `Discrete(2)`
- `rescue`: `Discrete(2)`
- `steer`: `Box(-1.0, 1.0, (1,), float32)`

## `observation_space`:

- `attachment`: `Discrete(10)`
- `attachment_time_left`: `Box(0.0, inf, (1,), float32)`
- `center_path`: `Box(-inf, inf, (3,), float32)`
- `center_path_distance`: `Box(-inf, inf, (1,), float32)`
- `distance_down_track`: `Box(-inf, inf, (1,), float32)`
- `energy`: `Box(0.0, inf, (1,), float32)`
- `front`: `Box(-inf, inf, (3,), float32)`
- `items_position`: `Sequence(Box(-inf, inf, (3,), float32), stack=False)`
- `items_type`: `Sequence(Discrete(7), stack=False)`
- `jumping`: `Discrete(2)`
- `karts_position`: `Sequence(Box(-inf, inf, (3,), float32), stack=False)`
- `max_steer_angle`: `Box(-1.0, 1.0, (1,), float32)`
- `paths_distance`: `Sequence(Box(0.0, inf, (2,), float32), stack=False)`
- `paths_end`: `Sequence(Box(-inf, inf, (3,), float32), stack=False)`
- `paths_start`: `Sequence(Box(-inf, inf, (3,), float32), stack=False)`
- `paths_width`: `Sequence(Box(0.0, inf, (1,), float32), stack=False)`
- `powerup`: `Discrete(11)`
- `shield_time`: `Box(0.0, inf, (1,), float32)`
- `skeed_factor`: `Box(0.0, inf, (1,), float32)`
- `velocity`: `Box(-inf, inf, (3,), float32)`

# Available Wrappers and Environment from PySTK2-Gymansium

## Environments

After importing `pystk2_gymnasium`, the following environments are available:

- `supertuxkart/full-v0` is the main environment containing complete
  observations. The observation and action spaces are both dictionaries with
  continuous or discrete variables (see below). The exact structure can be found
  using `env.observation_space` and `env.action_space`. The following options
  can be used to modify the environment:
    - `agent` is an `AgentSpec (see above)`
    - `render_mode` can be None or `human`
    - `track` defines the SuperTuxKart track to use (None for random). The full
      list can be found in `STKRaceEnv.TRACKS` after initialization with
      `initialize.initialize(with_graphics: bool)` has been called.
    - `num_kart` defines the number of karts on the track (3 by default)
    - `max_paths` the maximum number of the (nearest) paths (a track is made of
      paths) to consider in the observation state
    - `laps` is the number of laps (1 by default)
    - `difficulty` is the difficulty of the AI bots (lowest 0 to highest 2,
      default to 2)

Some environments are created using wrappers (see below for wrapper
documentation),
- `supertuxkart/simple-v0` (wrappers: `ConstantSizedObservations`) is a
  simplified environment with a fixed number of observations for paths
  (controlled by `state_paths`, default 5), items (`state_items`, default 5),
  karts (`state_karts`, default 5)
- `supertuxkart/flattened-v0` (wrappers: `ConstantSizedObservations`,
  `PolarObservations`, `FlattenerWrapper`) has observation and action spaces
  simplified at the maximum (only `discrete` and `continuous` keys)
- `supertuxkart/flattened_continuous_actions-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `OnlyContinuousActionsWrapper`, `FlattenerWrapper`) removes discrete actions
  (default to 0) so this is steer/acceleration only in the continuous domain
- `supertuxkart/flattened_multidiscrete-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `DiscreteActionsWrapper`, `FlattenerWrapper`) is like the previous one, but with
  fully multi-discrete actions. `acceleration_steps` and `steer_steps` (default
  to 5) control the number of discrete values for acceleration and steering
  respectively.
- `supertuxkart/flattened_discrete-v0` (wrappers: `ConstantSizedObservations`, `PolarObservations`, `DiscreteActionsWrapper`, `FlattenerWrapper`, `FlattenMultiDiscreteActions`) is like the previous one, but with fully
  discretized actions

The reward $r_t$ at time $t$ is given by

$$ r_{t} =  \frac{1}{10}(d_{t} - d_{t-1}) + (1 - \frac{\mathrm{pos}_t}{K})
\times (3 + 7 f_t) - 0.1 + 10 * f_t $$

where $d_t$ is the overall track distance at time $t$, $\mathrm{pos}_t$ the
position among the $K$ karts at time $t$, and $f_t$ is $1$ when the kart
finishes the race.

## Wrappers

Wrappers can be used to modify the environment.

### Constant-size observation

`pystk2_gymnasium.ConstantSizedObservations( env, state_items=5,
  state_karts=5, state_paths=5 )` ensures that the number of observed items,
karts and paths is constant. By default, the number of observations per category
is 5.

### Polar observations

`pystk2_gymnasium.PolarObservations(env)` changes Cartesian
coordinates to polar ones (angle in the horizontal plane, angle in the vertical plan, and distance) of all 3D vectors.

### Discrete actions

`pystk2_gymnasium.DiscreteActionsWrapper(env, acceleration_steps=5, steer_steps=7)` discretizes acceleration and steer actions (5 and 7 values respectively).

### Flattener (actions and observations)

This wrapper groups all continuous and discrete spaces together.

`pystk2_gymnasium.FlattenerWrapper(env)` flattens **actions and
observations**. The base environment should be a dictionary of observation
spaces. The transformed environment is a dictionary made with two entries,
`discrete` and `continuous` (if both continuous and discrete
observations/actions are present in the initial environment, otherwise it is
either the type of `discrete` or `continuous`). `discrete` is `MultiDiscrete`
space that combines all the discrete (and multi-discrete) observations, while
`continuous` is a `Box` space.

### Flatten multi-discrete actions

`pystk2_gymnasium.FlattenMultiDiscreteActions(env)` flattens a multi-discrete
action space into a discrete one, with one action per possible unique choice of
actions. For instance, if the initial space is $\{0, 1\} \times \{0, 1, 2\}$,
the action space becomes $\{0, 1, \ldots, 6\}$.
