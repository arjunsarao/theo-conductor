from schema import Task
import json
from scheduler import topological_sort
from traces import print_step_order


def main() -> None:
    response = json.load(open("../../examples/parallel.json"))
    x = Task.from_dict(response)
    sorted_steps = topological_sort(x)
    print_step_order(sorted_steps)


if __name__ == "__main__":
    main()
