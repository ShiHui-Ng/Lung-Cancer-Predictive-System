import yaml
from pipeline import MLPipeline

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = MLPipeline(config)

    # run pipeline (Triggers training + saving)
    result = pipeline.run()

    print("Training completed.")
    print(f"Best model: {result['name']}")
    print(f"Metrics: {result['metrics']}")

if __name__ == "__main__":
    main()

