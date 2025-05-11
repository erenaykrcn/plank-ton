import portfolio

def run(input_data,solver_params,extra_arguments):
  if 'evaluation_date' in extra_arguments:
    input_data['evaluation_date']=extra_arguments['evaluation_date']
  return portfolio.run(input_data)
