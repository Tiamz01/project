import logging
import os
from datetime import datetime

LOG_FILE_FORMAT=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_file=os.path.join(os.getcwd(),"logs",LOG_FILE_FORMAT)
os.makedirs(logs_file,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_file,LOG_FILE_FORMAT)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



if __name__=="__main__":
    logging.info('Logging has started')
   

