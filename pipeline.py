from kfp import compiler
from kfp.dsl import pipeline
from src.pipeline_components import (
    data_extraction, 
    data_preprocessing, 
    model_training, 
    model_evaluation
)


@pipeline(name="California Housing Pipeline")
def california_pipeline():
    # Extract data
    extract_task = data_extraction()
    
    # Preprocess data
    preprocess_task = data_preprocessing(
        input_data=extract_task.outputs["output_data"]
    )
    
    # Train model
    train_task = model_training(
        train_data=preprocess_task.outputs["train_data"]
    )
    
    # Evaluate model
    eval_task = model_evaluation(
        model=train_task.outputs["model"],
        test_data=preprocess_task.outputs["test_data"]
    )


# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=california_pipeline,
        package_path="components/california_housing_pipeline.yaml"
    )
    print("Pipeline YAML compiled successfully!")