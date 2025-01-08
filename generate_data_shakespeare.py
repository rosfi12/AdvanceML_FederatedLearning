import os
import json

def generate_mock_shakespeare_dataset(output_dir="./data/shakespeare"):
    os.makedirs(output_dir, exist_ok=True)

    # Example mock data
    users = [f"user_{i}" for i in range(10)]
    mock_data = {
        "users": users,
        "num_samples": [100 for _ in users],
        "user_data": {
            user: {
                "x": ["To be, or not to be: that is the question." * 5 for _ in range(100)],
                "y": [0 for _ in range(100)]  # Placeholder labels
            }
            for user in users
        }
    }

    # Save mock train and test datasets
    train_path = os.path.join(output_dir, "train_data.json")
    test_path = os.path.join(output_dir, "test_data.json")

    with open(train_path, "w") as train_file:
        json.dump(mock_data, train_file)

    with open(test_path, "w") as test_file:
        json.dump(mock_data, test_file)

    print(f"Mock Shakespeare dataset created at {output_dir}")


if __name__ == "__main__":
    generate_mock_shakespeare_dataset()
