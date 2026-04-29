# 配置类
class BaseConfig:
    MAX_INPUT_LEN = 40
    MAX_OUTPUT_LEN = 60
    MIN_RESPONSE_LEN = 2
    BATCH_SIZE = 16
    LEARNING_RATE = 0.003
    TEACHER_FORCING_RATIO = 0.5
    SAVE_PATH = "best_model.pth"

class FastConfig(BaseConfig):
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    EPOCHS = 60

class QualityConfig(BaseConfig):
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    EPOCHS = 80