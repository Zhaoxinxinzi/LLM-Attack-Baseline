import argparse
from custom_models import CustomSwitchTransformersForSequenceClassification

def print_model_parameters(model_path, num_labels):
    try:
        model = CustomSwitchTransformersForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=num_labels,
            trust_remote_code=True
        )
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(name)
    except Exception as e:
        print(f"Error loading model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Print model parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of labels for the classification task")
    args = parser.parse_args()

    print_model_parameters(args.model_path, args.num_labels)

if __name__ == "__main__":
    main()