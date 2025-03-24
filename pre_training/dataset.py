import torch

class Dataset(torch.utils.data.Dataset):
    """
    This class loads and preprocesses the given text data
    """
    def __init__(self, paths, tokenizer):
        """
        This function initialises the object. It takes the given paths and tokeniser.
        """
        self.paths = paths
        self.tokenizer = tokenizer
#        self.data = self.read_file(self.paths[0])
        self.current_file = 0
        self.offset = 0
#        self.remaining = len(self.data)
        
#         # get length
#         self.length = 0
#         for path in self.paths: 
#             #print(len(self.read_file(path)))
#             self.length += len(self.read_file(path))
        
        # code to read in all at once
        files = []
        for path in self.paths: files.append(self.read_file(path))
        self.data = []
        for file in files:
            for token in file:
                self.data.append(token)
        print(len(self.data))
        self.encodings = self.get_encodings(self.data)

    def __len__(self):
        """
        returns the length of the ds
        """
        #return self.length
        # return 1058750 # pre-calculated length of 10M data set
        return 10587561
    
    def read_file(self, path):
        """
        reads a given file
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encodings(self, lines_all):
        """
        Creates encodings for a given text input
        """
        # tokenise all text 
        batch = self.tokenizer(lines_all, max_length=128, padding='max_length', truncation=True)

        # Ground Truth
        labels = torch.tensor(batch['input_ids'])
        # Attention Masks
        mask = torch.tensor(batch['attention_mask'])

        # Input to be masked
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)

        # with a probability of 15%, mask a given word, leave out CLS, SEP and PAD
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        # assign token 4 (=MASK)
        input_ids[mask_arr] = 4
        
        return {'input_ids':input_ids, 'attention_mask':mask, 'labels':labels}

    def __getitem__(self, i):
        """
        returns item i
        Note: do not use shuffling for this dataset
        """
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.offset += len(self.data)
            self.current_file += 1
            # if we are at the end of the dataset, start over again
            if self.current_file == len(self.paths):
                self.current_file = 0
            # self.get_encodings(self.data)
            print("reading {}".format(self.paths[self.current_file]))
            self.data = self.read_file(self.paths[self.current_file])
            self.remaining = len(self.data)
        
        # reset offset when i is reset
        if i == 0:
            self.offset = 0
        
        self.remaining -= 1

        encodings = self.get_encodings(self.data[i - self.offset])

        return encodings 
