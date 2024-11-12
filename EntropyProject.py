import os
import numpy
from PIL import Image
from tabulate import tabulate

class EntropyCalculator():

    def __init__(self):

        self.working_directory = "./Pictures/"
        self.image_type = (".jpeg", ".jpg", ".png")
        self.sorted_size = []
        self.sorted_entropy = []
        self.output_array = []

        # Use list comprehension to get all files matching image_type in working_directory
        self.images = [file for file in os.listdir(self.working_directory) if file.endswith(self.image_type)]

    def run(self):

        for image_name in self.images:

            file_path = self.working_directory + image_name
    
            # Open image and convert to grayscale for entropy calculation
            image = Image.open(file_path).convert("L")
           
            # Get width and height of image and calculate total pixels
            width, height = image.size
            image_size = width * height
            self.sorted_size.append([image_name, int(image_size)])

            entropy_value = self.calculateEntropy(image)
            self.sorted_entropy.append([image_name, float(entropy_value)])
        
        # Sort the size and entropy arrays in ascending order
        self.sorted_size.sort(key=lambda x: x[1])
        self.sorted_entropy.sort(key=lambda x: x[1])

        self.outputResults()

        # Output number of duplicate files in each interval in size and entropy arrays
        print((f"\nDuplicates in Top 25: {self.calculateDuplicatesInInterval(0, 25)}\n"
                f"Duplicates in Middle 51: {self.calculateDuplicatesInInterval(25, 76)}\n"
                f"Duplicates in Bottom 25: {self.calculateDuplicatesInInterval(76, 101)}"))

        # Output number of duplicate files in each opposite interval in size and entropy arrays
        print((f"\nDuplicates in Top 25 Entropy/Bottom 25 Size: {self.calculateDuplicatesInOppositeInterval(0, 25)}\n"
                f"Duplicates in Bottom 25 Entropy/Top 25 Size: {self.calculateDuplicatesInOppositeInterval(76, 101)}"))

        # Output number of files within shift of each other in size and entropy arrays
        print((f"\nFiles w/ Position +5/-5 Apart in Sorted Lists: {self.calculateFilesInShift(5)}\n"
                f"Files w/ Position +10/-10 Apart in Sorted Lists: {self.calculateFilesInShift(10)}\n"
                f"Files w/ Position +25/-25 Apart in Sorted Lists: {self.calculateFilesInShift(25)}"))

    def calculateEntropy(self, image):
        
        # Convert image to numpy array
        img_array = numpy.array(image)

        # Flatten 2D array 
        pixel_values = img_array.flatten()

        # Calculate histogram of pixels
        hist, _ = numpy.histogram(pixel_values, bins=256, range=(0,256))

        # Calculate probabilities of each pixel
        probabilities = hist / hist.sum()

        # Calculate the entropy using entropy formula
        #
        # Adding numpy.finfo(float).eps to probabilities handles the case where the probability
        # is zero. It's the smallest positive integer that registers in the float type, something
        # like 10^-16. This way the program doesn't throw an error when taking numpy.log2(0).
        entropy = -numpy.sum(probabilities * numpy.log2(probabilities + numpy.finfo(float).eps))

        return entropy

    def outputResults(self):

        for i in range(0, len(self.sorted_entropy)):
            self.output_array.append([i+1] + self.sorted_entropy[i] + self.sorted_size[i])

        print(tabulate(self.output_array, headers=["Rank", "File Name", "Entropy", "File Name", "Size"], tablefmt="pretty"))

    def calculateDuplicatesInInterval(self, start, end):
        
        # Slice the sorted arrays on start and end, create new sorted arrays with only file_names
        entropy_interval = [file_name for (file_name, entropy_value) in self.sorted_entropy[start:end]]
        size_interval = [file_name for (file_name, entropy_value) in self.sorted_size[start:end]]

        # Compare the number of duplicate file names in the interval of the two arrays. Ideally, 
        # this would show some correlation, like: "the smallest files have the lowest entropy".
        return len(set(entropy_interval) & set(size_interval))

    def calculateDuplicatesInOppositeInterval(self, start, end):
        
        # Slice the sorted_entropy on start and end, create new sorted array with only file_names
        entropy_interval = [file_name for (file_name, entropy_value) in self.sorted_entropy[start:end]]

        # Since I'm only using this to compare top 25 entropy w/ bottom 25 size and vice versa, this
        # is the lazy way of getting it to work without actually finding [start:end]'s true opposite.
        if (start == 0):
            size_interval = [file_name for (file_name, entropy_value) in self.sorted_size[(len(self.sorted_size) - (end - start)):len(self.sorted_size)]]
        else:
            size_interval = [file_name for (file_name, entropy_value) in self.sorted_size[0:(end - start)]]

        # Compare the number of duplicate file names in the interval of the two arrays. Ideally, 
        # this would show some correlation, like: "the smallest files have the lowest entropy".
        return len(set(entropy_interval) & set(size_interval))

    def calculateFilesInShift(self, shift):

        file_counter = 0

        # Create new arrays with just the file names
        sorted_entropy_filenames = [file_name for (file_name, entropy_value) in self.sorted_entropy] 
        sorted_size_filenames = [file_name for (file_name, entropy_value) in self.sorted_size]

        # For each file, determine if its position in the size array is +/- shift away from its
        # position in the entropy array. For example, if we're looking at the file at index 44 in
        # the entropy array and shift is 5, determine if it's anywhere in positions 39-49 in the
        # size array.
        for i in range(0, len(sorted_entropy_filenames)):
            if (sorted_entropy_filenames[i] == sorted_size_filenames[i]):
                file_counter += 1

            maximum = min(i + 1 + shift, len(sorted_entropy_filenames))
            minimum = max(i - 1 - shift, -1)

            for j in range(i + 1, maximum):
                if (sorted_entropy_filenames[i] == sorted_size_filenames[j]):
                    file_counter += 1

            for k in range(i - 1, minimum, -1):
                if (sorted_entropy_filenames[i] == sorted_size_filenames[k]):
                    file_counter += 1

        return file_counter
                
        

if __name__ == "__main__":
    ec = EntropyCalculator()
    ec.run()
