import numpy as np
from skimage.transform import resize
from skimage.util import random_noise
from PIL import Image, ImageFont, ImageDraw
import string

def generate_handwritten_char_image(character, image_size=(28, 28), font_path='path/to/font.ttf', noise_level=0.1):
    """
    Generates a single image of a handwritten character.

    Parameters:
    - character: The character to generate.
    - image_size: The size of the output image (height, width).
    - font_path: Path to the .ttf font file to use for generating the character.
    - noise_level: The level of random noise to add to the image.

    Returns:
    - An image array of the handwritten character.
    """
    # Create a blank image with white background
    image = Image.new('L', image_size, 'white')
    draw = ImageDraw.Draw(image)

    # Load the font and calculate font size to fill the image
    font_size = int(image_size[0] * 0.8)
    font = ImageFont.truetype(font_path, font_size)

    # Get character size and position
    text_width, text_height = draw.textsize(character, font=font)
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2

    # Draw the character
    draw.text((x, y), character, font=font, fill='black')

    # Convert to numpy array
    image_array = np.array(image)

    # Resize to desired size
    image_array = resize(image_array, image_size, mode='constant', anti_aliasing=True)

    # Add random noise
    image_array = random_noise(image_array, var=noise_level**2)

    return image_array

def handwriting_recognition_data_generator(chars=string.ascii_letters + string.digits, batch_size=32, image_size=(28, 28), font_path='path/to/font.ttf'):
    """
    A generator that yields batches of handwritten character images and their labels.

    Parameters:
    - chars: A string containing the characters to generate.
    - batch_size: The number of images to yield in each batch.
    - image_size: The size of each image (height, width).
    - font_path: Path to the .ttf font file to use for generating characters.

    Yields:
    - A tuple (X_batch, y_batch) where X_batch is a batch of image data and y_batch are the corresponding labels.
    """
    while True:
        # Initialize batches
        X_batch = np.zeros((batch_size, *image_size), dtype=np.float32)
        y_batch = np.zeros((batch_size, len(chars)), dtype=np.int8)

        for i in range(batch_size):
            # Randomly select a character
            char = np.random.choice(list(chars))
            char_index = chars.index(char)

            # Generate image
            image = generate_handwritten_char_image(char, image_size, font_path)

            # Populate batches
            X_batch[i] = image
            y_batch[i, char_index] = 1

        yield X_batch, y_batch
