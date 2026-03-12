from models import sq_ast_mod
import sys
import os
import yaml
import pandas as pd
from datetime import datetime

if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = "configs/" + sys.argv[1] + ".yaml"
    else:
        config_path = "configs/default.yaml"

    with open(config_path, 'r') as f:
        config_file = yaml.safe_load(f)

    output_dir = config_file["output_dir"] + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_ind_csv_path = os.path.join(output_dir, "output_individual.csv")

    output_ind_df = pd.DataFrame(columns=["file_path", "sq_mos", "sq_noi", "sq_dis", "sq_col", "sq_loud"])

    sq_ast_pred, attention_flows, fbank_lengths, audio_files_txt = sq_ast_mod.sq_ast_fw(config_file)

    for i, (file_idx, file_path) in enumerate(audio_files_txt):
        output_ind_df.loc[i] = {
            "index": i,
            "file_path": file_path,
            "sq_mos": sq_ast_pred[i, 0],
            "sq_noi": sq_ast_pred[i, 1],
            "sq_dis": sq_ast_pred[i, 2],
            "sq_col": sq_ast_pred[i, 3],
            "sq_loud": sq_ast_pred[i, 4]
        }

    output_ind_df.to_csv(output_ind_csv_path, index=False)