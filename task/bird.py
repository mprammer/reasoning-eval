from numpy.random import random_integers
from datasets import load_dataset
from typing import Optional
from .base import Dataset
import logging
from sqlite3 import connect as sqlite3_connect
from os.path import realpath, abspath, join as path_join, exists, dirname
from os import mkdir, scandir
from requests import get as requests_get
from zipfile import ZipFile
from shutil import copyfile, copyfileobj

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BIRD(Dataset):
    def __init__(self, name: str = "lamini/bird_text_to_sql", split="train"):
        super().__init__(name, split)
        """
        The linked dataset provides a model input and a golden model output. The golden output is 
        mostly the same as the BIRD dataset's original golden output, while the input seems to have been
        generated in-house at Lamini using some schema-to-nl procedure.
        
        Beyond the dataset provided, we need to grab the original BIRD dataset to get the sqlite databases
        in order to run the queries to verify output correctness.
        WARNING: The BIRD dataset is a bit under 40GB of data once uncompressed.
        """
        # set up the dataset
        dataset_original = load_dataset(self.dataset_name)
        # as BIRD is almost 10k rows, we grab 30 examples to match the AIME size
        num_rows = 30
        sampled_query_indices = list(random_integers(0, len(dataset_original), num_rows))
        sampled_query_indices.sort()
        dataset = dataset_original[split].select(indices=sampled_query_indices)
        logging.debug(f"Dataset size: {len(dataset)}.")

        # setup BIRD
        # we use a top-level "resources" directory to hold the BIRD downloads
        default_downloaded_resources_directory = path_join(dirname(abspath(realpath(__file__))), "..", "resources")
        if not exists(default_downloaded_resources_directory):
            mkdir(default_downloaded_resources_directory)
        bird_dir = path_join(default_downloaded_resources_directory, "BIRD")
        if not exists(bird_dir):
            mkdir(bird_dir)

        # check if databases exist
        if not exists(path_join(bird_dir, "train_databases")):
            # outer databases directory doesn't exist, check for decompressed BIRD data
            if not exists(path_join(bird_dir, "train")):
                # if decompressed data doesn't exist, check for BIRD download.
                if not exists(path_join(bird_dir, "train.zip")):
                    # bird base download does not exist, downloading.
                    logging.info(f"Downloading BIRD dataset...")
                    bird_dataset_url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
                    with requests_get(bird_dataset_url, stream=True) as response:
                        with open(path_join(bird_dir, "train.zip"), 'wb') as downloaded_file:
                            copyfileobj(response.raw, downloaded_file)
                with ZipFile(path_join(bird_dir, "train.zip")) as bird_main_zip:
                    # the zipfile "train.zip" stores all data in a "train" directory
                    # and a stray __MACOSX directory
                    logging.info(f"Extracting main BIRD compressed archive...")
                    bird_main_zip.extract(bird_main_zip.getinfo("train"), bird_dir)
            # find the databases zip file in the BIRD directory
            if not exists(path_join(bird_dir, "train", "train_databases")):
                # something went wrong if we arrived here.
                logging.critical(f"Unable to find BIRD databases archive! Unable to validate queries!")
            with ZipFile(path_join(bird_dir, "train", "train_databases.zip")) as bird_databases_zip:
                # this zip file contains all database files in a "train_databases" directory
                # and another __MACOSX dir
                logging.info(f"Extracting BIRD datasets compressed archive...")
                bird_databases_zip.extract(bird_databases_zip.getinfo("train_databases"), bird_dir)
            # while we're here, copy the gold answers file to the datasets file
            copyfile(path_join(bird_dir, "train", "train_gold.sql"), path_join(bird_dir, "train_databases", "gold.txt"))
        # find database files
        bird_databases = [x.name for x in scandir(path_join(bird_dir, "train_databases")) if x.is_dir()]

        # extract database associations from gold.txt
        query_index_to_database = {}
        with open(path_join(bird_dir, "train_databases", "gold.txt")) as f:
            for i, line in enumerate(f):
                if i in sampled_query_indices:
                    _raw_query, a_database = line.rsplit("\t", maxsplit=1)
                    query_index_to_database[i] = a_database
                    if len(query_index_to_database.keys()) == len(sampled_query_indices):
                        break
        sql_to_database = {}
        for local_idx, sampled_query_index in enumerate(sampled_query_indices):
            sql_to_database[dataset['output'][local_idx]] = query_index_to_database[sampled_query_index]
        # finalize
        logging.debug(f"BIRD Setup done. Found {len(bird_databases)} databases.")
        self.data = dataset
        self.bird_dir = bird_dir
        self.sampled_query_indices = sampled_query_indices
        self.bird_databases = bird_databases
        self.sql_to_database = sql_to_database



    def __iter__(self):
        """
        Iterate through the dataset.
        """
        for example in self.data:
            yield example["input"], example["output"]

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.data)

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract output SQL.
        The output should begin with SELECT and either extend to the end of the response or end with a semicolon.
        """
        if not response:
            return None

        first_select = response.find("SELECT")
        last_semicolon = response.rfind(";")
        query_extract_start = 0 if first_select == -1 else first_select
        query_extract_end = len(response) if last_semicolon == -1 else last_semicolon
        return response[query_extract_start:query_extract_end]

    def is_correct(self, completion, answer):
        """
        Check if the completion is correct based on the answer.
        :param completion: The generated answer from the model.
        :param answer: The ground truth answer.
        :return: True if the completion is correct, False otherwise.
        """
        # extract sql from the answer
        # the gold answer should not need to be processed
        sql_response = self.extract_answer(completion)
        # sql_gold = self.extract_answer(completion)
        # use the answer to get the sqlite database
        database_name = self.sql_to_database[answer]
        database_path = path_join(self.bird_dir, "train_databases", database_name, f"{database_name}.sqlite")
        database_connection = sqlite3_connect(database_path)
        # execute the gold and provided answers
        result_gold = database_connection.execute(answer).fetchall()
        result_response = database_connection.execute(sql_response).fetchall()
        # compare answers
        if len(result_gold) != len(result_response):
            return False
        else:
            for a_tuple_gold, a_tuple_response in zip(result_gold, result_response):
                # tuple equality should work to verify the outputs
                if a_tuple_gold != a_tuple_response:
                    return False
        return True
