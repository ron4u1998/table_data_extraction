from PIL import Image
img = Image.open("input_images/input_2.jpg")
img2 = img.crop((9.9399323e+01, 9.4084760e+02, 1.5456571e+03, 1.7484852e+03))
img2.save("cropped_output_images/output2.jpg")


