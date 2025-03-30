class Dataset(object):
    def __init__(self, name: str, split: str = "train"):
        self.dataset_name = name
        self.split = split
        
    def __iter__(self):
        """
        Iterate through the dataset.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def __len__(self):
        """
        Get the length of the dataset.
        """ 
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self):
        return f"Dataset(name={self.name})"
    
    def extract_answer(self, response: str):
        """
        Extract the answer from the response string.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def is_correct(self, completion: str, answer: str):
        """
        Check if the completion is correct based on the answer.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    