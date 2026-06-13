def compute_reward(obs, action, next_obs, done, info):
    components = {}
    # component: distance_penalty
    component_0 = float(-10.0 * abs(obs[0]))
    component_0 = max(min(component_0, 0.0), -100.0)
    components['distance_penalty'] = float(component_0)
    # component: velocity_penalty
    component_1 = float(-1.0 * (obs[2]**2 + obs[3]**2 + obs[5]**2))
    component_1 = max(min(component_1, 0.0), -100.0)
    components['velocity_penalty'] = float(component_1)
    # component: angle_penalty
    component_2 = float(-5.0 * abs(obs[4]))
    component_2 = max(min(component_2, 0.0), -50.0)
    components['angle_penalty'] = float(component_2)
    # component: fuel_efficiency
    component_3 = float(-0.1 * float(action != 0))
    component_3 = max(min(component_3, 0.0), -10.0)
    components['fuel_efficiency'] = float(component_3)
    # component: touchdown_bonus
    component_4 = float(10.0 * float(obs[6] == 1.0 and obs[7] == 1.0 and abs(obs[0]) < 0.1 and abs(obs[2]) < 0.1 and abs(obs[3]) < 0.1 and abs(obs[4]) < 0.1))
    component_4 = max(min(component_4, 100.0), 0.0)
    components['touchdown_bonus'] = float(component_4)
    # component: terminal
    component_5 = float(-100.0 * float(done and not (obs[6] == 1.0 and obs[7] == 1.0 and abs(obs[0]) < 0.1 and abs(obs[2]) < 0.1 and abs(obs[3]) < 0.1 and abs(obs[4]) < 0.1)))
    component_5 = max(min(component_5, 0.0), -1000.0)
    components['terminal'] = float(component_5)
    total_reward = component_0 + component_1 + component_2 + component_3 + component_4 + component_5
    total_reward = max(min(total_reward, 1000.0), -1000.0)
    return float(total_reward), components
