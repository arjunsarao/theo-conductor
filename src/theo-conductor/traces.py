"""
This file should provide the logging functionality for the runner. It should allow for logging of errors, warnings, and informational messages. The logging should be configurable to allow for different levels of verbosity and should support logging to both the console and to a file.
"""
from task import Step

def print_step_order(scheduled_steps: list[list[Step]]) -> None:
    """
    Print the order of steps in the scheduled workflow.
    """
    for layer_index, layer in enumerate(scheduled_steps):
        print(f"Layer {layer_index + 1}:")
        for step in layer:
            print("\t" + step.step_id)