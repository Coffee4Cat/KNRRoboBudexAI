#         "Wind efficiency"
#         "Solar power"
#         "Fiber optics"
#         "Temperature"
#         "Popilation density"

import torch
import numpy as np

def rule_more_renewable(xa, xb):
    # Weights
    w_wind = 0.02
    w_solar = 3.0

    val_a_wind = xa[:, 2] * w_wind
    val_a_solar = xa[:, 3] * w_solar
    val_b_wind = xb[:, 2] * w_wind
    val_b_solar = xb[:, 3] * w_solar
    score_a = torch.max(val_a_wind, val_a_solar)
    score_b = torch.max(val_b_wind, val_b_solar)

    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_more_fiber_optics(xa, xb):
    # Higher fiber optics is better
    score_a = xa[:, 4]
    score_b = xb[:, 4]
    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r*w

def rule_heating_region(xa, xb):
    # For temperature above 15°C, weight is zero.
    # For lower temperatures, the lower the temperature, the higher the weight.
    temp_a = xa[:, 5]
    temp_b = xb[:, 5]
    # Weight: zero if temp > 15, else (15 - temp)
    weight_a = torch.where(temp_a > 15, 0.0, 15 - temp_a)
    weight_b = torch.where(temp_b > 15, 0.0, 15 - temp_b)
    r = torch.sign(temp_b - temp_a)  # positive if a is lower
    w = torch.abs(weight_a - weight_b)
    return r * w

def rule_facilities(xa, xb):
    # Higher population density is better, but only worth it above 100
    pop_a = xa[:, 6]
    pop_b = xb[:, 6]
    score_a = torch.where(pop_a > 100, xa[:, 6], 0.0)
    score_b = torch.where(pop_b > 100, xb[:, 6], 0.0)
    r = torch.sign(pop_a - pop_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_possible_heat(xa, xb):
    # Calculate possible heat score: only for temperatures below 20°C and scaled by population density
    score_a = torch.where(xa[:, 5] < 20, 20 - xa[:, 5], 0.0) * xa[:,6] / 100
    score_b = torch.where(xb[:, 5] < 20, 20 - xb[:, 5], 0.0) * xb[:,6] / 100
    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_good_solar(xa, xb, lower=4.0, upper=20.0):
    """
    Solar score: zero below `lower`, capped at `upper`.
    Values between are scaled linearly.
    """
    def score(x):
        s = torch.clamp(x[:, 3], min=lower, max=upper)  # clamp to [lower, upper]
        s = s - lower  # shift so lower bound is zero
        return s

    score_a = score(xa)
    score_b = score(xb)

    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    w_solar = 3.0
    return r * w * w_solar

def rule_high_wind(xa, xb, lower=50.0, upper=250.0):
    """
    Wind score: zero below `lower`, capped at `upper`.
    Values between are scaled linearly.
    """
    def score(x):
        s = torch.clamp(x[:, 2], min=lower, max=upper)
        s = s - lower
        return s

    score_a = score(xa)
    score_b = score(xb)

    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    w_wind = 0.02
    return r * w * w_wind


def rule_renewables_need_cooling(xa, xb):
    # if sign different, capacity_score more important
    renewable_score = rule_more_renewable(xa, xb)
    capacity_score = rule_possible_heat(xa, xb)
    score = torch.sign(capacity_score) * torch.abs((torch.sign(capacity_score) - torch.sign(renewable_score)))
    return score


def total_rule_score(xa, xb, N=5):
    """
    Pick N rules (randomly) using tensor operations and combine their scores on GPU.
    """
    rules = [
        rule_more_renewable,
        rule_more_fiber_optics,
        rule_heating_region,
        rule_facilities,
        rule_possible_heat,
        rule_good_solar,
        rule_high_wind,
        rule_renewables_need_cooling
    ]

    num_rules = len(rules)
    device = xa.device  # assumes xa and xb are already on GPU

    # Randomly select 4 unique indices using torch
    indices = torch.randperm(num_rules, device=device)[:N]

    # Compute scores for chosen rules
    scores = torch.stack([rules[i](xa, xb) for i in indices])

    normalized_scores = torch.clamp(scores, -5.0, 5.0) / 5.0

    total_score = normalized_scores.sum(dim=0)
    return total_score

print("test")

test_data_a = torch.tensor([
    [100,100, 100, 4.814, 0.392, 17.0, 89], 
    [100,100, 50, 3.840, 0.214, 6.4, 18],   
    [100,100, 130, 4.327, 0.303, 11.7, 38]    
])

# Rounded test data B
test_data_b = torch.tensor([
    [100,100, 150, 4.327, 0.303, 11.7, 38],  
    [100,100, 14, 4.127, 0.353, 13.7, 138],  
    [100,100, 100, 4.627, 0.283, 8.7, 1]    
])

print(f'Rule more renewable: {rule_more_renewable(test_data_a, test_data_b)}')
print(f'Rule more fiber optics: {rule_more_fiber_optics(test_data_a, test_data_b)}')
print(f'Rule heating region: {rule_heating_region(test_data_a, test_data_b)}')
print(f'Rule facilities: {rule_facilities(test_data_a, test_data_b)}')
print(f'Rule possible heat: {rule_possible_heat(test_data_a, test_data_b)}')
print(f'Rule good solar: {rule_good_solar(test_data_a, test_data_b)}')
print(f'Rule high wind: {rule_high_wind(test_data_a, test_data_b)}')
print(f'Renewables need cooling: {rule_renewables_need_cooling(test_data_a, test_data_b)}')
print(f'Total rule score: {total_rule_score(test_data_a, test_data_b)}')

