import os
import shutil
from torch import save, load

SAVED_MODELS_PATH = "Data/models"


def get_iteration_to_restore(restart=False):
    base_path = f"{SAVED_MODELS_PATH}/iteratoins"
    if not os.path.exists(base_path):
        return 0
    if restart:
        shutil.rmtree(base_path)
        return 0
    saved_iter = os.listdir(base_path)
    if len(saved_iter) == 0:
        return 0
    saved_iter = [int(iter) for iter in saved_iter if iter.isdigit()]
    saved_iter.sort(reverse=True)
    restore_iter = 0
    for iter in saved_iter:
        if os.path.exists(f"{base_path}/{iter}/state"):
            restore_iter = iter
            break
    return restore_iter


def save_running_state(iteration,
                       best_validation_score,
                       best_learning_rate,
                       best_iteration,
                       model_state_dict,
                       optimizer_state_dict,
                       scheduler_state_dict):
    print(f"Save running status for iteration {iteration}")
    base_path = f"{SAVED_MODELS_PATH}/iteratoins"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    prev_iters = os.listdir(f"{SAVED_MODELS_PATH}/iteratoins")
    path = f"{base_path}/{str(iteration)}"
    os.mkdir(path)
    state = {
        "iteration": iteration,
        "best_validation_score": best_validation_score,
        "best_learning_rate": best_learning_rate,
        "best_iteration": best_iteration,
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict
    }
    save(state, f"{path}/state")
    for prev_iter in prev_iters:
        shutil.rmtree(f"{base_path}/{prev_iter}")


def restore_iteration_state(iteration):
    path = f"{SAVED_MODELS_PATH}/iteratoins/{iteration}/state"
    if not os.path.exists(path):
        return None
    state = load(path)
    return state
