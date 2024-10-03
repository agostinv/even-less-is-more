import yaml
import argparse

def add_args(parser):
    parser.add_argument('model_name', type=str, help='Name of the model config is being generated for')
    parser.add_argument('--average-budget', type=float, default=0.10, 
                        help='Total cache budget, averaged across all layers. Alternatively, this is the cache budget ' \
                            + 'for all layers in uniform applications.')
    parser.add_argument('--num-layers', type=int, default=32, 
                        help='Number of layers to generate cache budget config for, if not specified, default is 32 layers ' \
                            + 'based on typical 7B sizings.')
    parser.add_argument('--budget-split-type', type=str, default='uniform', choices=['uniform', 'linear', 'power-law'],
                        help='Type of budget split to use for cache budgeting. Default is uniform (typical LESS).')
    parser.add_argument('--budget-minimum', type=float, default=0.01, 
                        help='Minimum cache budget for any layer, default is 0.01. Used to infer behavior for linear ' \
                            + 'and power-law budget splits.')
    parser.add_argument('--force-minimum', action='store_true', 
                        help='Force minimum budget to be same, solves an issue with power-law distributions but inflates the average.')

def get_budget_split(average_budget, budget_split_type, budget_minimum, force_minimum, num_layers, li):
    if budget_split_type == 'uniform':
        return {
            'fixed_budget': average_budget * 0.05,
            'heavy_budget': average_budget * 0.475,
            'recent_budget': average_budget * 0.475
        }, average_budget
    
    elif budget_split_type == 'linear':
        mod_budget = budget_minimum + 2 * (average_budget - budget_minimum) * (1 - (li / (num_layers - 1)))
        return {
            'fixed_budget': mod_budget * 0.05,
            'heavy_budget': mod_budget * 0.475,
            'recent_budget': mod_budget * 0.475
        }, mod_budget
    
    elif budget_split_type == 'power-law':
        # exponent chosen somewhat arbitrarily, for high exponents (e.g. -3) average gets inflated
        minimum = 0 if not force_minimum else budget_minimum
        normalization_factor = sum([(i + 1) ** -0.7 for i in range(num_layers)])

        # attempt to generate something of a dynamic scaling factor to compensate
        # normally, power law distributions have no well-defined mean for exponents > -2, but
        # this particular distribution model is bounded, so we can model average_we_want = scaling * E(x)
        # and then multiply both sides by the number of layers to get the total
        unscaled_total = sum([(average_budget - budget_minimum) * ((i + 1) ** -0.7) / normalization_factor for i in range(num_layers)])
        scaling_factor = (average_budget * num_layers) / unscaled_total

        mod_budget = max(minimum, scaling_factor * (average_budget - budget_minimum) * ((li + 1) ** -0.7) / normalization_factor)
        return {
            'fixed_budget': mod_budget * 0.05,
            'heavy_budget': mod_budget * 0.475,
            'recent_budget': mod_budget * 0.475
        }, mod_budget


def generate_yaml(model_name, num_layers, average_budget, budget_split_type, budget_minimum, force_minimum):
    data = {
        'model': model_name,
        'layers': {}
    }

    total_budget = 0
    budget_min = average_budget
    for i in range(num_layers):
        budget_split, budget_step = get_budget_split(average_budget, budget_split_type, budget_minimum, force_minimum, num_layers, i)
        total_budget += budget_step

        if budget_step < budget_min:
            budget_min = budget_step

        layer_key = f'layer_{i}'
        data['layers'][layer_key] = budget_split

    actual_average_budget = round(total_budget / num_layers, 4)
    print(f'Actual average budget for {model_name}: {actual_average_budget}')
    if not (args.budget_split_type == "power-law" and args.force_minimum):
        assert actual_average_budget == average_budget, \
            f'Total budget ({actual_average_budget}) does not match expected, average budget ({average_budget}) for all layers.'
    
    budget_min = round(budget_min, 4)
    print(f'Minimum budget for {model_name}: {budget_min}')
    if not budget_split_type == "power-law":
        assert budget_min == (budget_minimum if not budget_split_type == "uniform" else average_budget), \
            f'Actual minimum budget ({budget_min}) does not match expected minimum budget ({budget_minimum}).'
    elif not force_minimum or budget_min > budget_minimum:
        print(f"WARNING: power-law distribution minimum budget is not guaranteed to be actual minimum. Double check to ensure " \
              + "the difference is acceptable.")

    with open(f'{model_name.lower().replace(" ", "_")}_{budget_split_type.replace("-", "_")}_{average_budget}_{budget_minimum}_budget.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML file for model cache budget.')
    add_args(parser)
    args = parser.parse_args()

    print(f'Generating cache budget YAML for {args.model_name} with {args.num_layers} layers and average, average budget of {args.average_budget}.')
    print(f'Using {args.budget_split_type} budget split with minimum budget of {args.budget_minimum}.')
    if args.budget_split_type == 'power-law' and args.force_minimum:
        print(f'WARNING: Enforcing minimum budget for power-law distribution. This may inflate the average budget.')

    # pretty literal assertions
    assert args.average_budget > 0 and args.budget_minimum > 0
    assert args.num_layers > 0
    assert args.average_budget >= args.budget_minimum

    generate_yaml(args.model_name, args.num_layers, args.average_budget, args.budget_split_type, args.budget_minimum, args.force_minimum)