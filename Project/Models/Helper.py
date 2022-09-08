import os
from torch import save, load

SAVED_MODELS_PATH = "Data/models"


def save_running_state(model_name,
                       iteration,
                       best_learning_rate,
                       best_bleu_score,
                       best_wer_score,
                       best_validation_loss,
                       model_state_dict,
                       optimizer_state_dict,
                       scheduler_state_dict):
    if not os.path.exists(f"{SAVED_MODELS_PATH}/{model_name}"):
        os.mkdir(f"{SAVED_MODELS_PATH}/{model_name}")
    state = {
        "iteration": iteration,
        "best_learning_rate": best_learning_rate,
        "best_bleu_score": best_bleu_score,
        "best_wer_score": best_wer_score,
        "best_validation_loss": best_validation_loss,
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "scheduler": scheduler_state_dict
    }
    save(state, f"{SAVED_MODELS_PATH}/{model_name}/state")


def restore_iteration_state(model_name):
    base_path = f"{SAVED_MODELS_PATH}/{model_name}"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(f"{base_path}/state"):
        return None
    state = load(f"{base_path}/state")
    return state
