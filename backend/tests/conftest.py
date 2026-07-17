import os

# Inject dummy environment variables to bypass Pydantic settings validation during tests
os.environ["FURIGANA_MODEL_LOCAL_DIR"] = "/mock/model/dir"
os.environ["FURIGANA_LOG_LEVEL"] = "DEBUG"
