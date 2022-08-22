import os
from torch import save, load

SAVED_MODELS_PATH = "Data/models"


def save_running_state(iteration,
                       best_validation_score,
                       best_learning_rate,
                       best_iteration,
                       model_state_dict,
                       optimizer_state_dict,
                       scheduler_state_dict):
    print(f"Save running status for iteration {iteration}")
    if not os.path.exists(SAVED_MODELS_PATH):
        print(SAVED_MODELS_PATH)
        os.mkdir(SAVED_MODELS_PATH)
    state = {
        "iteration": iteration,
        "best_validation_score": best_validation_score,
        "best_learning_rate": best_learning_rate,
        "best_iteration": best_iteration,
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict
    }
    save(state, f"{SAVED_MODELS_PATH}/state")


def restore_iteration_state():
    path = f"{SAVED_MODELS_PATH}/state"
    if not os.path.exists(path):
        return None
    state = load(path)
    return state
