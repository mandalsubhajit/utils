# 1. Save
# new model directory will be created if non-existent
model.save('/path/to/model/directory/')

# 2. Load
from keras.models import load_model
model = load_model('/path/to/model/directory/')
