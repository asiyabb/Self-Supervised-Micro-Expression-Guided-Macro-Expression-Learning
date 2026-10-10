import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Dataset Paths (Combining CASME II and MMEW Micro for micro-expressions)
CASME_MICRO_ROOT = "/home/ab3379/work/data/CASME/Cropped"
MMEW_MICRO_ROOT = "/home/ab3379/work/data/MMEW_Final/Micro_Expression"
MMEW_MACRO_ROOT = "/home/ab3379/work/data/MMEW_Final/Macro_Expression"

# Excel Metadata Files
CASME_CODING_EXCEL = os.path.join(PROJECT_ROOT, "CASME2-coding-20140508.xlsx")
CASME_OBJ_EXCEL = os.path.join(PROJECT_ROOT, "CASME2-ObjectiveClasses.xlsx")
MMEW_MICRO_EXCEL = os.path.join(PROJECT_ROOT, "MMEW_Micro_Exp.xlsx")

# Subject Split Ratios & Hyperparameters
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_HEIGHT = 112
IMG_WIDTH = 112
NUM_FRAMES_MICRO = 16
NUM_FRAMES_MACRO = 32
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
EMBED_DIM = 256
TEMPERATURE = 0.5