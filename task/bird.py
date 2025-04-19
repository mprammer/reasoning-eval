from numpy.random import random_integers
from datasets import load_dataset
from typing import Optional
from .base import Dataset
import logging
from sqlite3 import connect as sqlite3_connect
from os.path import realpath, abspath, join as path_join, exists, dirname
from os import mkdir, scandir, system
from zipfile import ZipFile

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
        bird_dir = realpath(path_join(default_downloaded_resources_directory, "BIRD"))
        if not exists(bird_dir):
            mkdir(bird_dir)
        gold_sql_path = realpath(path_join(bird_dir, "train", "train_gold.sql"))
        databases_path = realpath(path_join(bird_dir, "train_databases"))

        # check if databases exist
        if not exists(databases_path) or not exists(gold_sql_path):
            # missing files, proceed to check for all dependency files
            # checking for original archive
            bird_dataset_dl_path = realpath(path_join(bird_dir, "train.zip"))
            if not exists(bird_dataset_dl_path):
                # bird base download does not exist, downloading.
                logging.info(f"Downloading BIRD dataset...")
                bird_dataset_url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
                # the BIRD download seems to intermittently fail
                # using wget avoids the hassle of retrying
                system(f"wget {bird_dataset_url} -O {bird_dataset_dl_path}")
                logging.info(f"BIRD dataset downloaded to: {bird_dataset_dl_path}")
            # checking for gold solutions
            if not exists(gold_sql_path):
                with ZipFile(path_join(bird_dir, "train.zip")) as bird_databases_zip:
                    bird_databases_zip.extract(bird_databases_zip.getinfo("train/train_gold.sql"), bird_dir)
                    logging.info(f"Extracted BIRD gold sql to: {bird_dir}/train/train_gold.sql")
            # checking for databases archive
            if not exists(path_join(bird_dir, "train_databases.zip")):
                with ZipFile(path_join(bird_dir, "train.zip")) as bird_databases_zip:
                    logging.info(f"Extracting BIRD databases compressed archive...")
                    bird_databases_zip.extract(bird_databases_zip.getinfo("train/train_databases.zip"), bird_dir)
                    logging.info(f"Extracted BIRD compressed databases to: {bird_dir}/train/train_databases.zip")
            # checking for databases directory
            if not exists(databases_path):
                with ZipFile(path_join(bird_dir, "train", "train_databases.zip")) as bird_databases_zip:
                    logging.info(f"Extracting BIRD databases compressed databases...")
                    bird_databases_zip.extractall(bird_dir)
                    logging.info(f"Extracted BIRD compressed databases to: {bird_dir}/train_databases/")

        # find database files
        bird_databases = [x.name for x in scandir(path_join(bird_dir, "train_databases")) if x.is_dir()]

        # extract database associations from gold.txt
        query_index_to_database = {}
        with open(gold_sql_path) as f:
            for i, line in enumerate(f):
                if i in sampled_query_indices:
                    _raw_query, a_database = line.rsplit("\t", maxsplit=1)
                    query_index_to_database[i] = a_database.strip()
                    if len(query_index_to_database.keys()) == len(sampled_query_indices):
                        logging.info(f"Finished parsing \"{gold_sql_path}\" by line: {i}")
                        break
        sql_to_database = {}
        for local_idx, sampled_query_index in enumerate(sampled_query_indices):
            sql_to_database[dataset['output'][local_idx]] = query_index_to_database[sampled_query_index]

        # finalize
        logging.info(f"BIRD Setup done. Found {len(bird_databases)} databases: {', '.join(bird_databases)}")
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

        last_select = response.rfind("SELECT")
        last_semicolon = response.rfind(";")
        query_extract_start = 0 if (last_select == -1) else last_select
        query_extract_end = len(response) if (last_semicolon <= last_select) else (last_semicolon + 1)
        clean_newlines = ' '.join(response[query_extract_start:query_extract_end].splitlines())
        return clean_newlines

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
        database_path = realpath(path_join(self.bird_dir, "train_databases", database_name, f"{database_name}.sqlite"))
        database_connection = sqlite3_connect(database_path)
        # execute the gold and provided answers
        result_gold = database_connection.execute(answer).fetchall()
        try:
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
        except BaseException as e:
            # logging.info(f"Model produced an output that caused an exception.")
            # logging.info(f"gold: {answer}")
            # logging.info(f"response: {sql_response}")
            return False

