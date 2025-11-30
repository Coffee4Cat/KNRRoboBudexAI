#         "Wind efficiency"
#         "Solar power"
#         "Fiber optics"
#         "Temperature"
#         "Popilation density"

import torch
import numpy as np

def rule_more_renewable(xa, xb):
    # Sum wind efficiency and solar power
    w_wind = 3.0
    w_solar = 0.02
    score_a = xa[:, 0] * w_wind + xa[:, 1] * w_solar
    score_b = xb[:, 0] * w_wind + xb[:, 1] * w_solar
    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r*w

def rule_more_fiber_optics(xa, xb):
    # Higher fiber optics is better
    score_a = xa[:, 2]
    score_b = xb[:, 2]
    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r*w

def rule_heating_region(xa, xb):
    # For temperature above 15°C, weight is zero.
    # For lower temperatures, the lower the temperature, the higher the weight.
    temp_a = xa[:, 3]
    temp_b = xb[:, 3]
    # Weight: zero if temp > 15, else (15 - temp)
    weight_a = torch.where(temp_a > 15, 0.0, 15 - temp_a)
    weight_b = torch.where(temp_b > 15, 0.0, 15 - temp_b)
    r = torch.sign(temp_b - temp_a)  # positive if a is lower
    w = torch.abs(weight_a - weight_b)
    return r * w

def rule_facilities(xa, xb):
    # Higher population density is better, but only worth it above 100
    pop_a = xa[:, 4]
    pop_b = xb[:, 4]
    score_a = torch.where(pop_a > 100, xa[:, 4], 0.0)
    score_b = torch.where(pop_b > 100, xb[:, 4], 0.0)
    r = torch.sign(pop_a - pop_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_possible_heat(xa, xb):
    # Calculate possible heat score: only for temperatures below 20°C and scaled by population density
    score_a = torch.where(xa[:, 3] < 20, 20 - xa[:, 3], 0.0) * xa[:,4] / 100
    score_b = torch.where(xb[:, 3] < 20, 20 - xb[:, 3], 0.0) * xb[:,4] / 100
    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_good_solar(xa, xb, lower=4.0, upper=20.0):
    """
    Solar score: zero below `lower`, capped at `upper`.
    Values between are scaled linearly.
    """
    def score(x):
        s = torch.clamp(x[:, 1], min=lower, max=upper)  # clamp to [lower, upper]
        s = s - lower  # shift so lower bound is zero
        return s

    score_a = score(xa)
    score_b = score(xb)

    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r * w

def rule_high_wind(xa, xb, lower=50.0, upper=500.0):
    """
    Wind score: zero below `lower`, capped at `upper`.
    Values between are scaled linearly.
    """
    def score(x):
        s = torch.clamp(x[:, 0], min=lower, max=upper)
        s = s - lower
        return s

    score_a = score(xa)
    score_b = score(xb)

    r = torch.sign(score_a - score_b)
    w = torch.abs(score_a - score_b)
    return r * w


def total_rule_score(xa, xb):
    scores = [
        rule_more_renewable(xa, xb),
        rule_more_fiber_optics(xa, xb),
        rule_heating_region(xa, xb),
        rule_facilities(xa, xb),
        rule_possible_heat(xa, xb),
        rule_good_solar(xa, xb),
        rule_high_wind(xa, xb)
    ]

    # Cap each score at 5 (both positive and negative)
    capped_scores = [torch.clamp(s, min=-5.0, max=5.0) for s in scores]

    # Sum the capped scores
    total_score = sum(capped_scores)
    return total_score


print("test")

test_data_a = torch.tensor([
    [0.410, 4.814, 0.392, 17.0, 89],   # Above avg for most
    [0.168, 3.840, 0.214, 6.4, 18],    # Below avg for most
    [0.289, 4.327, 0.303, 11.7, 38]    # Near avg
])

# Rounded test data B
test_data_b = torch.tensor([
    [0.289, 4.327, 0.303, 11.7, 38],   # Near avg
    [0.349, 4.127, 0.353, 13.7, 138],  # Mixed case
    [0.239, 4.627, 0.283, 8.7, 1]    # Mixed case
])

print(f'Rule more renewable: {rule_more_renewable(test_data_a, test_data_b)}')
print(f'Rule more fiber optics: {rule_more_fiber_optics(test_data_a, test_data_b)}')
print(f'Rule heating region: {rule_heating_region(test_data_a, test_data_b)}')
print(f'Rule facilities: {rule_facilities(test_data_a, test_data_b)}')
print(f'Rule possible heat: {rule_possible_heat(test_data_a, test_data_b)}')
print(f'Rule good solar: {rule_good_solar(test_data_a, test_data_b)}')
print(f'Rule high wind: {rule_high_wind(test_data_a, test_data_b)}')
print(f'Total rule score: {total_rule_score(test_data_a, test_data_b)}')

