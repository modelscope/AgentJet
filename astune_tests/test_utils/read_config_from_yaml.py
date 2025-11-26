def read_yaml_config(file_path: str) -> dict:
    import yaml

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config
