import pandas as pd

def main():
    train_data = pd.read_csv("InjectionMolding_Train.csv")
    test_data = pd.read_csv("InjectionMolding_Test.csv")

    print(train_data)

if __name__ == "__main__":
    main()