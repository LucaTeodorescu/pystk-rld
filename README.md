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