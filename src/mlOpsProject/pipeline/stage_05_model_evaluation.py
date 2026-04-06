from mlOpsProject.components.model_evaluation import ModelEvaluation
from mlOpsProject.config.configuration import ConfigurationManager
from mlOpsProject import config, logger

STAGE_NAME="Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.log_into_mlflow()
        except Exception as e:
            raise e
        

if __name__=="__main__":
    try:
        logger.info(f">>>>>>> {STAGE_NAME} started <<<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f"{'>>'*20} {STAGE_NAME} completed {'<<'*20}")
    except Exception as e:
        logger.exception(e)
        raise e

