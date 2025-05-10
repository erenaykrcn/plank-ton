#########  THIS FILE WILL BE REPLACED  #########

input_file_name = "input.json"


import json
with open(input_file_name) as f:
  input_data = json.load(f)

data = input_data["data"]

# Optional extra parameters
if "extra_arguments" in input_data:
    extra_arguments = input_data['extra_arguments']
else:
    extra_arguments = {}

if "solver_params" in input_data:
    solver_params = input_data['solver_params']
else:
    solver_params = {}


import main

result = main.run(data, solver_params, extra_arguments)

print(result)

################################################
#########    DO NOT UNTIL FROM HERE    #########
################################################