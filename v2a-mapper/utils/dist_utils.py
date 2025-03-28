import os
from logging import Logger

local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1

