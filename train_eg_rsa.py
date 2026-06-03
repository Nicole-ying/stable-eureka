from eg_rsa import EGRSARunner


if __name__ == "__main__":
    runner = EGRSARunner(config_path="./configs/eg_rsa_minimal.yml")
    runner.run()
