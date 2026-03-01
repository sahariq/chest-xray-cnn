import kagglehub


def main() -> None:
    covid_path = kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")
    print("Path to dataset files (COVID19/Pneumonia):", covid_path)

    pneumonia_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files (Chest X-Ray Pneumonia):", pneumonia_path)


if __name__ == "__main__":
    main()
