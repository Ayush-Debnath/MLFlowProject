
from re import S

from mlOpsProject import logger
from mlOpsProject.components import data_ingestion
from mlOpsProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlOpsProject.pipeline.stage_02_data_validation import DataValidationPipeline
from mlOpsProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME="Data Ingestion stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion=DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Validation stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_validation=DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Data Transformation stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj=DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e