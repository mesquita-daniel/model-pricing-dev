# Model Pricing

Repo created to study the problem of model pricing. The model statement is better
defined on this [medium article](https://daniel-mesquita.medium.com/a-very-simplified-approach-to-ml-models-pricing-4af251226779).

## Installation

````
pip install git+https://github.com/mesquita-daniel/model-pricing-dev
````

## Usage

````python
from model_pricing import *

#First define a scenario
scenario = Scenario(badrate=0.1,fee=0.4)

# Expected profit from a model with given AUC
expected_profit(auc=0.8,scenario=scenario)

# Expected profit increase when switching models
expected_profit_increase(base_auc=0.5,new_model_auc=0.8,scenario=scenario)

# Minimum AUC to change justify a change from a base model
critical_auc(scenario=scenario,base_auc=0.5)
````