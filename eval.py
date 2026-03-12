from models import sq_ast_mod
import yaml

if __name__ == "__main__":
    with open("configs/default.yaml", 'r') as f:
        config_file = yaml.safe_load(f)

    sq_ast_mod.sq_ast_fw(config_file)