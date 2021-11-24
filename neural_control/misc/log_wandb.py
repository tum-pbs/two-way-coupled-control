import os
import json
import wandb
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Log results to Weights and Biases")
    parser.add_argument("run_path", help="path to run folder")
    run_path = parser.parse_args().run_path
    run_name = run_path.split('/')[-2]
    # Load inputs json file
    with open(run_path + '/inputs.json', 'r') as f:
        inp = json.load(f)
    # Load test metrics json file
    tests = os.listdir(run_path + '/tests/')
    os.environ["WANDB_DIR"] = run_path
    os.environ["WANDB_MODE"] = "run"  # Run wandb online so we can sync with previous run
    wandb.init(project="neural_controller_translation_only" if inp["translation_only"] else "neural_controller", name=run_name)
    wandb.config.update(inp)
    for test in tests:
        try:
            with open(run_path + f'/tests/{test}/metrics.json', 'r') as f:
                metrics = json.load(f)
        except:
            print(f"Did not find metrics for test {test}")
            continue
        metrics_labels = list(metrics.keys())
        metrics_values = list(metrics.values())
        metrics_labels, metrics_values = zip(*sorted(zip(metrics_labels, metrics_values,), key=lambda metrics: isinstance(metrics[1], list)))
        # Start a new wandb run
        # Log inputs
        # Log metrics
        for metric_label, metric_values in zip(metrics_labels, metrics_values):
            if isinstance(metric_values, list):
                for value in metric_values: wandb.log({test + "_" + metric_label: value})
            else:
                wandb.log({test + "_" + metric_label: metric_values})
    # Append run id to inputs file
    with open(run_path + '/inputs.json', 'w') as f:
        inp["id"] = wandb.run.id
        json.dump(inp, f, indent="    ")
    # Finish logging
    wandb.run.name = run_name
    wandb.run.save()
    wandb.finish()
