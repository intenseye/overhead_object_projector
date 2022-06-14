import subprocess
import os
import sys
import json

params = dict(batch_size="32", loss_function_reg="mse", tuning_method="loss", fixed_partition_seed="True",
              validation_ratio="0.010", test_ratio="0.988", zero_mean_enabled="False", use_mixed_precision="False",
              max_epoch="2500", init_learning_rate="8.8809958930846e-05", scheduler_type="one_cycle", betas_0="0.9",
              betas_1="0.999", weight_decay="1.6291723470742454e-06", eps_adam="3.305798423244983e-11",
              init_w_normal="False")


def wandb_sweep_runner():
    modified_env = os.environ.copy()
    modified_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    config_to_run = {}
    for i in range(1, len(sys.argv)):
        args = sys.argv[i].strip("-").split("=")
        config_to_run.update({args[0]: args[1]})
    params.update(config_to_run)

    with open(os.path.join('temp_params.py'), 'w') as convert_file:
        convert_file.write('param_sweep = ')
        convert_file.write(json.dumps(params))

    input_txt_path = '/home/poyraz/intenseye/input_outputs/crane_simulation/inputs_outputs_w_roll_dev_non_norm.txt'
    projection_axis = 'y'
    driver = 'wandb'

    args = [sys.executable, "projection_trainer.py",
            "--driver", driver,
            "--input_txt_path", input_txt_path,
            "--projection_axis", projection_axis]
    subprocess.call(args, stdout=None, stderr=None, shell=False, env=modified_env)


if __name__ == "__main__":
    wandb_sweep_runner()
