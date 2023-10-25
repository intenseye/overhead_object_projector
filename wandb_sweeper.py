import subprocess
import os
import sys
import json

params = dict(network_size="m", batch_size="128", loss_function_reg="mse", test_model_mode="best_accuracy_model",
              fixed_partition_seed="True", use_mixed_precision="False", max_epoch="5000",
              init_learning_rate="0.01002192776248166", scheduler_type="one_cycle", betas_0="0.9",
              betas_1="0.999", weight_decay="6.686846385772325e-07", eps_adam="8.394178078518644e-11",
              activation="relu")


def wandb_sweep_runner():
    """
    Sweeps the parameter and runs the training.
    """
    modified_env = os.environ.copy()
    modified_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    config_to_run = {}
    for i in range(1, len(sys.argv)):
        args = sys.argv[i].strip("-").split("=")
        config_to_run.update({args[0]: args[1]})
    params.update(config_to_run)

    with open(os.path.join('./temp_params/temp_params_m.py'), 'w') as convert_file:
        convert_file.write('param_sweep = ')
        convert_file.write(json.dumps(params))

    # Input txt and distance map can be generated by sample_generator.py
    driver = 'wandb'
    settings_path = r'./settings/settings_1.ini'

    args = [sys.executable, "projection_trainer.py",
            "--driver", driver,
            "--settings_path", settings_path]
    subprocess.call(args, stdout=None, stderr=None, shell=False, env=modified_env)


if __name__ == "__main__":
    wandb_sweep_runner()
