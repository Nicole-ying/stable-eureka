def compute_reward(obs, action, next_obs, done, info):
    components = {}
    # component: distance_reward
    component_0 = float(-1.0 * np.sqrt(next_obs[0]**2 + next_obs[1]**2))
    component_0 = max(min(component_0, 0.0), -5.0)
    components['distance_reward'] = float(component_0)
    # component: velocity_penalty
    component_1 = float(-0.5 * np.sqrt(next_obs[2]**2 + next_obs[3]**2))
    component_1 = max(min(component_1, 0.0), -5.0)
    components['velocity_penalty'] = float(component_1)
    # component: angle_penalty
    component_2 = float(-2.0 * abs(next_obs[4]))
    component_2 = max(min(component_2, 0.0), -5.0)
    components['angle_penalty'] = float(component_2)
    # component: fuel_efficiency
    component_3 = float(-0.5 * float(action == 2))
    component_3 = max(min(component_3, 0.0), -1.0)
    components['fuel_efficiency'] = float(component_3)
    # component: terminal
    component_4 = float(float(done) * (100.0 if (next_obs[6] == 1.0 and next_obs[7] == 1.0 and np.sqrt(next_obs[2]**2 + next_obs[3]**2) < 0.1 and abs(next_obs[4]) < 0.1 and np.sqrt(next_obs[0]**2 + next_obs[1]**2) < 0.1) else -50.0) + (0.1 * float(not done)))
    component_4 = max(min(component_4, 500.0), -500.0)
    components['terminal'] = float(component_4)
    total_reward = component_0 + component_1 + component_2 + component_3 + component_4
    total_reward = max(min(total_reward, 1000.0), -1000.0)
    return float(total_reward), components
