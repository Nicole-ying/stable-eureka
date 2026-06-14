def compute_reward(obs, action, next_obs, done, info):
    components = {}
    # component: shaping
    component_0 = float(2.0 * np.exp(-2.0 * np.sqrt(obs[0]**2 + obs[1]**2)))
    component_0 = max(min(component_0, 1000.0), -1000.0)
    components['shaping'] = float(component_0)
    # component: velocity_penalty
    component_1 = float(-0.4 * (abs(obs[2]) + abs(obs[3])))
    component_1 = max(min(component_1, 1000.0), -1000.0)
    components['velocity_penalty'] = float(component_1)
    # component: angle_penalty
    component_2 = float(-0.5 * (abs(obs[4]) + 0.5 * abs(obs[5])))
    component_2 = max(min(component_2, 1000.0), -1000.0)
    components['angle_penalty'] = float(component_2)
    # component: fuel_efficiency
    component_3 = float(-0.15 * float(action == 2) - 0.05 * float(action == 1 or action == 3) - 0.02)
    component_3 = max(min(component_3, 1000.0), -1000.0)
    components['fuel_efficiency'] = float(component_3)
    # component: terminal
    component_4 = float(500.0 * float(done and obs[6] > 0.5 and obs[7] > 0.5 and abs(obs[0]) < 0.3 and abs(obs[1]) < 0.3 and abs(obs[2]) < 0.5 and abs(obs[3]) < 0.5 and abs(obs[4]) < 0.3) - 200.0 * float(done and not (abs(obs[0]) < 0.3 and abs(obs[1]) < 0.3 and abs(obs[2]) < 0.5 and abs(obs[3]) < 0.5 and abs(obs[4]) < 0.3 and obs[6] > 0.5 and obs[7] > 0.5)))
    component_4 = max(min(component_4, 1000.0), -1000.0)
    components['terminal'] = float(component_4)
    total_reward = component_0 + component_1 + component_2 + component_3 + component_4
    total_reward = max(min(total_reward, 1000.0), -1000.0)
    return float(total_reward), components
