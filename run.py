import heligym
from gym.wrappers.normalize import NormalizeObservation
import stable_baselines3

heli = heligym.HeliForwardFlight()
heli = NormalizeObservation(heli)
model = stable_baselines3.PPO(
    policy='MlpPolicy',
    env=heli,
    tensorboard_log='log'
)

while True:
    #train
    model.learn(999)

    #test
    end = False
    obs = heli.reset()
    while not end:
        action, _ = model.predict(obs)
        obs, _, end, _ = heli.step(action)
        heli.render()