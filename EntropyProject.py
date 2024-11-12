import os
import numpy
from PIL import Image
from tabulate import tabulate

class EntropyCalculator():

    def __init__(self):

        self.working_directory = "./"
        self.image_type = (".jpeg", ".jpg", ".png")
        self.sorted_size = []
        self.sorted_entropy = []
        self.output_array = []
        self.images = [file for file in os.listdir(self.working_directory) if file.endswith(self.image_type)]

    def run(self):

        for image_name in self.images:
    
            # open image and convert to grayscale for entropy calculation
            image = Image.open(image_name).convert("L")
            
            width, height = image.size
            image_size = width * height
            self.sorted_size.append([image_name, int(image_size)])

            entropy_value = self.calculateEntropy(image)
            self.sorted_entropy.append([image_name, float(entropy_value)])
        
        self.sorted_size.sort(key=lambda x: x[1])
        self.sorted_entropy.sort(key=lambda x: x[1])

        self.outputResults()

    def calculateEntropy(self, image):
        
        # convert image to numpy array
        img_array = numpy.array(image)

        # flatten 2D array 
        pixel_values = img_array.flatten()

        # calculate histogram of pixels
        hist, _ = numpy.histogram(pixel_values, bins=256, range=(0,256))

        # calculate probabilities of each pixel
        probabilities = hist / hist.sum()

        # calculate the entropy using entropy formula
        entropy = -numpy.sum(probabilities * numpy.log2(probabilities + numpy.finfo(float).eps))

        return entropy

    def outputResults(self):

        for i in range(0, len(self.sorted_entropy)):
            self.output_array.append([i+1] + self.sorted_entropy[i] + self.sorted_size[i])

        print(tabulate(self.output_array, headers=["Rank", "File Name", "Entropy", "File Name", "Size"], tablefmt="pretty"))


if __name__ == "__main__":
    ec = EntropyCalculator()
    ec.run()
